package examples

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
)

// namedStore mimics a Store backed by an HTTP SQL gateway: every row travels as
// a JSON object keyed by column name, so two result columns with the same name
// collapse into one, and numbers come back as JSON numbers.
type namedStore struct {
	db *sql.DB
}

func (s *namedStore) Dialect() agent.Dialect { return agent.SQLite }

func (s *namedStore) Exec(ctx context.Context, q string, args ...any) (int64, error) {
	res, err := s.db.ExecContext(ctx, q, args...)
	if err != nil {
		return 0, err
	}
	n, _ := res.RowsAffected()
	return n, nil
}

func (s *namedStore) Query(ctx context.Context, q string, args ...any) (agent.Rows, error) {
	rows, err := s.db.QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	cols, err := rows.Columns()
	if err != nil {
		return nil, err
	}
	for i, c := range cols {
		cols[i] = postgresColumnName(c)
	}
	out := &namedRows{cols: cols, pos: -1}
	for rows.Next() {
		vals := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, err
		}
		obj := map[string]any{}
		for i, c := range cols {
			if b, ok := vals[i].([]byte); ok {
				obj[c] = string(b)
			} else {
				obj[c] = vals[i]
			}
		}
		raw, _ := json.Marshal(obj)
		var decoded map[string]any
		_ = json.Unmarshal(raw, &decoded)
		out.rows = append(out.rows, decoded)
	}
	return out, rows.Err()
}

// postgresColumnName names an unaliased result column the way Postgres does:
// a function call after its outermost function ("coalesce", "count"), any other
// expression "?column?". SQLite instead uses the expression text, which is
// unique and would hide a collision.
func postgresColumnName(c string) string {
	if i := strings.IndexByte(c, '('); i > 0 {
		return strings.ToLower(strings.TrimSpace(c[:i]))
	}
	if strings.ContainsAny(c, " +-*/") {
		return "?column?"
	}
	if i := strings.LastIndexByte(c, '.'); i >= 0 {
		return c[i+1:]
	}
	return c
}

type namedRows struct {
	cols []string
	rows []map[string]any
	pos  int
}

func (r *namedRows) Next() bool   { r.pos++; return r.pos < len(r.rows) }
func (r *namedRows) Err() error   { return nil }
func (r *namedRows) Close() error { return nil }

func (r *namedRows) Scan(dest ...any) error {
	if len(dest) != len(r.cols) {
		return fmt.Errorf("scan: %d destinations for %d columns", len(dest), len(r.cols))
	}
	row := r.rows[r.pos]
	for i, d := range dest {
		v := row[r.cols[i]]
		switch p := d.(type) {
		case *string:
			if v == nil {
				*p = ""
			} else {
				*p = fmt.Sprint(v)
			}
		case *int64:
			switch x := v.(type) {
			case float64:
				*p = int64(x)
			case string:
				n, err := strconv.ParseInt(x, 10, 64)
				if err != nil {
					return err
				}
				*p = n
			}
		case *float64:
			if x, ok := v.(float64); ok {
				*p = x
			}
		default:
			return fmt.Errorf("scan: unsupported destination %T", d)
		}
	}
	return nil
}

// TestNameKeyedStoreUsage runs a conversation and its usage summary through a
// store that keys rows by column name, which is what an HTTP gateway returns.
func TestNameKeyedStoreUsage(t *testing.T) {
	e := setup(t)
	e.cfg.Store = &namedStore{db: e.db}
	ctx := context.Background()
	client, err := agent.NewClient(ctx, e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, err := client.Agent(ctx, "", agent.AgentOptions{})
	if err != nil {
		t.Fatal(err)
	}
	e.llm.push(toolCall("c1", "get_order", `{"order_id":"O"}`), text("done"))
	if _, err := a.Chat(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	sum, err := client.Usage(ctx, a.SessionID())
	if err != nil {
		t.Fatal(err)
	}
	recs, err := client.ListUsage(ctx, a.SessionID())
	if err != nil {
		t.Fatal(err)
	}
	var want agent.UsageSummary
	for _, r := range recs {
		want.Calls++
		want.Usage.PromptTokens += r.Usage.PromptTokens
		want.Usage.CompletionTokens += r.Usage.CompletionTokens
		want.Cost.Total += r.Cost.Total
		want.Cost.Output += r.Cost.Output
	}
	if sum.Calls != 2 || sum.Calls != want.Calls ||
		sum.Usage.PromptTokens != want.Usage.PromptTokens ||
		sum.Usage.CompletionTokens != want.Usage.CompletionTokens ||
		abs(sum.Cost.Total-want.Cost.Total) > 1e-9 || abs(sum.Cost.Output-want.Cost.Output) > 1e-9 {
		t.Fatalf("summary %+v, want %+v", sum, want)
	}
	ms, err := client.LatestMessages(ctx, a.SessionID(), 10)
	if err != nil || len(ms) != 4 {
		t.Fatalf("%d messages, %v", len(ms), err)
	}
}
