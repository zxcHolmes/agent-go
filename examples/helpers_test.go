package examples

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

// reply is one scripted assistant turn, streamed as SSE.
type reply struct {
	msg       string        // assistant message JSON
	delay     time.Duration // pause between content chunks
	hangAfter int           // hang after this many content chunks (-1 = never)
	httpErr   int           // respond with this HTTP status and body instead
	streamErr string        // send this error object inside a 200 stream instead
	body      string
	prompt    int    // prompt_tokens to report (default 1000)
	finish    string // finish_reason to report (default stop / tool_calls)
}

// fakeLLM is a scripted, streaming OpenAI-compatible server.
type fakeLLM struct {
	mu       sync.Mutex
	requests []map[string]json.RawMessage
	replies  []reply
	hung     chan struct{} // receives when a reply starts hanging
}

func (f *fakeLLM) handler(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	var req map[string]json.RawMessage
	_ = json.Unmarshal(body, &req)
	f.mu.Lock()
	f.requests = append(f.requests, req)
	if len(f.replies) == 0 {
		f.mu.Unlock()
		http.Error(w, `{"error":{"message":"no scripted reply"}}`, http.StatusBadRequest)
		return
	}
	rp := f.replies[0]
	f.replies = f.replies[1:]
	f.mu.Unlock()
	if rp.httpErr != 0 {
		http.Error(w, rp.body, rp.httpErr)
		return
	}
	if rp.streamErr != "" {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprintf(w, "data: {\"error\":%s}\n\n", rp.streamErr)
		return
	}

	var m struct {
		Content   *string          `json:"content"`
		ToolCalls []map[string]any `json:"tool_calls"`
	}
	_ = json.Unmarshal([]byte(rp.msg), &m)
	w.Header().Set("Content-Type", "text/event-stream")
	fl := w.(http.Flusher)
	send := func(v any) {
		b, _ := json.Marshal(v)
		fmt.Fprintf(w, "data: %s\n\n", b)
		fl.Flush()
	}
	hang := func() {
		f.hung <- struct{}{}
		<-r.Context().Done()
	}
	var pieces []string
	if m.Content != nil {
		rs := []rune(*m.Content)
		for i := 0; i < len(rs); i += 3 {
			pieces = append(pieces, string(rs[i:min(i+3, len(rs))]))
		}
	}
	for i, p := range pieces {
		if i == rp.hangAfter {
			hang()
			return
		}
		send(map[string]any{"id": "resp_1", "choices": []any{map[string]any{"index": 0, "delta": map[string]any{"content": p}}}})
		time.Sleep(rp.delay)
	}
	if rp.hangAfter >= 0 && rp.hangAfter >= len(pieces) {
		hang()
		return
	}
	for i, tc := range m.ToolCalls {
		tc["index"] = i
		send(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{"tool_calls": []any{tc}}}}})
	}
	finish := "stop"
	if len(m.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	if rp.finish != "" {
		finish = rp.finish
	}
	send(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}})
	prompt := rp.prompt
	if prompt == 0 {
		prompt = 1000
	}
	fmt.Fprintf(w, `data: {"choices":[],"usage":{"prompt_tokens":%d,"completion_tokens":100,"total_tokens":%d,"prompt_tokens_details":{"cached_tokens":%d},"completion_tokens_details":{"reasoning_tokens":20}}}`+"\n\ndata: [DONE]\n\n", prompt, prompt+100, min(800, prompt))
}

func (f *fakeLLM) push(msgs ...string) {
	for _, m := range msgs {
		f.pushReply(reply{msg: m, hangAfter: -1})
	}
}

func (f *fakeLLM) pushReply(r reply) {
	f.mu.Lock()
	f.replies = append(f.replies, r)
	f.mu.Unlock()
}

func (f *fakeLLM) request(i int) []json.RawMessage {
	f.mu.Lock()
	defer f.mu.Unlock()
	var msgs []json.RawMessage
	_ = json.Unmarshal(f.requests[i]["messages"], &msgs)
	return msgs
}

func toolCall(id, method, params string) string {
	args, _ := json.Marshal(fmt.Sprintf(`{"jsonrpc":"2.0","method":%q,"params":%s,"id":1}`, method, params))
	return fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":%q,"type":"function","function":{"name":"json_rpc","arguments":%s}}]}`, id, args)
}

func text(s string) string { return fmt.Sprintf(`{"role":"assistant","content":%q}`, s) }

type env struct {
	db     *sql.DB
	store  agent.Store
	client *agent.Client
	llm    *fakeLLM
	cfg    agent.Config // tests tweak it before newAgent
}

type orderParams struct {
	OrderID string `json:"order_id" desc:"Order id"`
}

func (p orderParams) Validate() error {
	if p.OrderID == "" {
		return errors.New("order_id is required")
	}
	return nil
}

func setup(t *testing.T, methods ...agent.Method) *env {
	t.Helper()
	db, err := sql.Open("sqlite", "file:"+filepath.Join(t.TempDir(), "t.db")+"?_pragma=busy_timeout(5000)")
	if err != nil {
		t.Fatal(err)
	}
	db.SetMaxOpenConns(1)
	t.Cleanup(func() { db.Close() })
	store := agent.NewSQLStore(db, agent.SQLite)
	f := &fakeLLM{hung: make(chan struct{}, 4)}
	srv := httptest.NewServer(http.HandlerFunc(f.handler))
	t.Cleanup(srv.Close)
	methods = append(methods, agent.NewMethod("get_order", func(ctx context.Context, c *agent.Call, p orderParams) (any, error) {
		return map[string]any{"order_id": p.OrderID, "user": c.Value("user_id")}, nil
	}, agent.MethodDoc{Description: "Get an order"}))
	cfg := agent.Config{
		Store: store, Billing: agent.Pricing{Input: 100, Output: 1000, CacheRead: 10},
		BaseURL: srv.URL + "/v1", APIKey: "k", Model: "fake", ContextLength: 100000, MaxOutputTokens: 1000,
		SystemPrompt: "sys", RPCDoc: "docs", Methods: methods,
		StaleAfter: time.Second, MaxRetries: -1,
	}
	ctx := context.Background()
	client, err := agent.NewClient(ctx, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agent.NewClient(ctx, cfg); err != nil { // init is idempotent
		t.Fatal(err)
	}
	return &env{db: db, store: store, client: client, llm: f, cfg: cfg}
}

func (e *env) newAgent(t *testing.T, sessionID string) *agent.Agent {
	t.Helper()
	a, err := e.client.Agent(context.Background(), sessionID, agent.AgentOptions{
		ContextParams: map[string]any{"user_id": "u1"},
		Override:      func(cfg *agent.Config) { *cfg = e.cfg },
	})
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func roles(ms []agent.Message) string {
	var r []string
	for _, m := range ms {
		r = append(r, m.Role)
	}
	return strings.Join(r, ",")
}

func statuses(ms []agent.Message) string {
	var r []string
	for _, m := range ms {
		r = append(r, string(m.Status))
	}
	return strings.Join(r, ",")
}

// waitFor polls cond until it holds or the test times out.
func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timed out waiting for %s", what)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
