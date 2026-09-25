package examples

import (
	"bytes"
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

// fakeLLM is a scripted OpenAI-compatible server.
type fakeLLM struct {
	mu       sync.Mutex
	requests []map[string]json.RawMessage
	replies  []string // assistant message JSON, consumed in order
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
	msg := f.replies[0]
	f.replies = f.replies[1:]
	f.mu.Unlock()
	finish := "stop"
	if strings.Contains(msg, "tool_calls") {
		finish = "tool_calls"
	}
	fmt.Fprintf(w, `{"id":"resp_1","model":"fake","choices":[{"index":0,"message":%s,"finish_reason":%q}],
		"usage":{"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100,"prompt_tokens_details":{"cached_tokens":800},"completion_tokens_details":{"reasoning_tokens":20}}}`, msg, finish)
}

func (f *fakeLLM) push(msgs ...string) {
	f.mu.Lock()
	f.replies = append(f.replies, msgs...)
	f.mu.Unlock()
}

func toolCall(id, method, params string) string {
	args, _ := json.Marshal(fmt.Sprintf(`{"jsonrpc":"2.0","method":%q,"params":%s,"id":1}`, method, params))
	return fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":%q,"type":"function","function":{"name":"json_rpc","arguments":%s}}]}`, id, args)
}

func text(s string) string { return fmt.Sprintf(`{"role":"assistant","content":%q}`, s) }

type env struct {
	store agent.Store
	llm   *fakeLLM
	srv   *httptest.Server
	cfg   agent.Config
}

type orderParams struct {
	OrderID string `json:"order_id"`
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
	ctx := context.Background()
	if err := agent.Init(ctx, store); err != nil {
		t.Fatal(err)
	}
	if err := agent.Init(ctx, store); err != nil { // idempotent
		t.Fatal(err)
	}
	f := &fakeLLM{}
	srv := httptest.NewServer(http.HandlerFunc(f.handler))
	t.Cleanup(srv.Close)
	methods = append(methods, agent.Method{
		Name:        "get_order",
		Description: "Get an order",
		Handler: agent.Typed(func(ctx context.Context, c *agent.Call, p orderParams) (any, error) {
			return map[string]any{"order_id": p.OrderID, "user": c.Value("user_id")}, nil
		}),
	})
	return &env{store: store, llm: f, srv: srv, cfg: agent.Config{
		BaseURL: srv.URL + "/v1", APIKey: "k", Model: "fake", ContextLength: 100000, MaxOutputTokens: 1000,
		SystemPrompt: "sys", RPCDoc: "docs", ContextParams: map[string]any{"user_id": "u1"},
		Store: store, Billing: agent.Pricing{Input: 100, Output: 1000, CacheRead: 10}, Methods: methods,
	}}
}

func (e *env) newAgent(t *testing.T, sessionID string) *agent.Agent {
	t.Helper()
	cfg := e.cfg
	cfg.SessionID = sessionID
	a, err := agent.New(context.Background(), cfg)
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

func TestChatToolLoop(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	sid, err := agent.CreateSession(ctx, e.store, map[string]any{"owner": "u1"})
	if err != nil {
		t.Fatal(err)
	}
	a := e.newAgent(t, sid)
	e.llm.push(
		toolCall("c1", "get_order", `{}`),                 // invalid params -> error back to model
		toolCall("c2", "get_order", `{"order_id":"O-1"}`), // ok
		text("Order O-1 found."),
	)
	res, err := a.Chat(ctx, "where is my order?")
	if err != nil {
		t.Fatal(err)
	}
	if res.StopReason != agent.StopCompleted || res.Status != agent.StatusIdle || res.Reply() != "Order O-1 found." {
		t.Fatalf("result %+v", res)
	}
	if got := roles(res.Messages); got != "user,assistant,tool,assistant,tool,assistant" {
		t.Fatal(got)
	}
	if c := res.Messages[2].Content; !strings.Contains(c, `"code":-32602`) || !strings.Contains(c, "order_id is required") {
		t.Fatalf("invalid params not reported: %s", c)
	}
	if c := res.Messages[4].Content; c != `{"jsonrpc":"2.0","id":1,"result":{"order_id":"O-1","user":"u1"}}` {
		t.Fatalf("tool result: %s", c)
	}

	// Prefix cache: each request's messages must be a byte-exact prefix of the next.
	var prev []json.RawMessage
	for i, req := range e.llm.requests {
		var msgs []json.RawMessage
		_ = json.Unmarshal(req["messages"], &msgs)
		for j := range prev {
			if !bytes.Equal(prev[j], msgs[j]) {
				t.Fatalf("request %d message %d changed:\n%s\n%s", i, j, prev[j], msgs[j])
			}
		}
		prev = msgs
	}
	var sys struct{ Content string }
	_ = json.Unmarshal(prev[0], &sys)
	if !strings.Contains(sys.Content, "get_order") || !strings.Contains(sys.Content, "docs") {
		t.Fatalf("system prompt: %s", sys.Content)
	}
	if !bytes.Contains(e.llm.requests[0]["tools"], []byte(`"name":"json_rpc"`)) {
		t.Fatal("tool missing")
	}

	// Usage & billing: 3 calls x (200 uncached in, 800 cached, 100 out).
	sum, err := a.Usage(ctx)
	if err != nil {
		t.Fatal(err)
	}
	wantCredits := 3 * (200*100.0/1e6 + 800*10.0/1e6 + 100*1000.0/1e6)
	if sum.Calls != 3 || sum.Usage.CachedTokens != 2400 || sum.Usage.ReasoningTokens != 60 || abs(sum.Cost.Total-wantCredits) > 1e-9 {
		t.Fatalf("usage %+v", sum)
	}
	recs, _ := agent.ListUsage(ctx, e.store, sid)
	if len(recs) != 3 || !strings.Contains(string(recs[0].RawUsage), "cached_tokens") {
		t.Fatalf("records %+v", recs)
	}

	// Pagination.
	all, _ := a.LatestMessages(ctx, 100)
	latest, _ := a.LatestMessages(ctx, 2)
	if len(all) != 6 || roles(latest) != "tool,assistant" || latest[1].ID != all[5].ID {
		t.Fatalf("latest %s", roles(latest))
	}
	before, _ := a.MessagesBefore(ctx, all[3].ID, 2)
	after, _ := a.MessagesAfter(ctx, all[3].ID, 10)
	if len(before) != 2 || before[0].ID != all[1].ID || len(after) != 2 || after[0].ID != all[4].ID {
		t.Fatal("before/after")
	}
	if _, err := a.MessagesBefore(ctx, "missing", 1); !errors.Is(err, agent.ErrMessageNotFound) {
		t.Fatal(err)
	}

	// Resume the session with a fresh agent.
	e.llm.push(text("again"))
	res, err = e.newAgent(t, sid).Chat(ctx, "thanks")
	if err != nil || res.Reply() != "again" {
		t.Fatal(res, err)
	}
	var msgs []json.RawMessage
	_ = json.Unmarshal(e.llm.requests[3]["messages"], &msgs)
	if len(msgs) != 8 { // system + 6 history + new user
		t.Fatalf("resumed request has %d messages", len(msgs))
	}
}

func TestConfirmation(t *testing.T) {
	var refunded []string
	e := setup(t, agent.Method{
		Name:           "refund",
		RequireConfirm: true,
		Handler: agent.Typed(func(ctx context.Context, c *agent.Call, p orderParams) (any, error) {
			refunded = append(refunded, p.OrderID)
			return "ok", nil
		}),
	})
	ctx := context.Background()
	a := e.newAgent(t, "")

	e.llm.push(toolCall("c1", "refund", `{"order_id":"O-1"}`))
	res, err := a.Chat(ctx, "refund O-1")
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != agent.StatusWaitingConfirmation || res.StopReason != agent.StopWaitingConfirmation || len(res.PendingCalls) != 1 {
		t.Fatalf("%+v", res)
	}
	if st, _ := a.Status(ctx); st != agent.StatusWaitingConfirmation {
		t.Fatal(st)
	}
	if _, err := a.Chat(ctx, "hello?"); !errors.Is(err, agent.ErrWaitingConfirmation) {
		t.Fatalf("want ErrWaitingConfirmation, got %v", err)
	}
	if _, err := a.Confirm(ctx, agent.Approve("nope")); !errors.Is(err, agent.ErrCallNotPending) {
		t.Fatalf("want ErrCallNotPending, got %v", err)
	}
	if len(refunded) != 0 {
		t.Fatal("ran before approval")
	}

	e.llm.push(text("Refunded."))
	res, err = a.Confirm(ctx, agent.Approve(res.PendingCalls[0].ID))
	if err != nil || res.Reply() != "Refunded." || len(refunded) != 1 || res.Status != agent.StatusIdle {
		t.Fatalf("%+v %v %v", res, err, refunded)
	}

	// Rejection goes back to the model as an error.
	e.llm.push(toolCall("c2", "refund", `{"order_id":"O-2"}`), text("OK, not refunding."))
	res, _ = a.Chat(ctx, "refund O-2")
	res, err = a.Confirm(ctx, agent.Reject(res.PendingCalls[0].ID, "too large"))
	if err != nil || len(refunded) != 1 {
		t.Fatal(err, refunded)
	}
	if c := res.Messages[0].Content; !strings.Contains(c, `"code":-32001`) || !strings.Contains(c, "too large") {
		t.Fatalf("rejection: %s", c)
	}
	if _, err := a.Confirm(ctx, agent.Approve("x")); !errors.Is(err, agent.ErrNoPendingCalls) {
		t.Fatal(err)
	}
}

func TestStopAndContinue(t *testing.T) {
	started := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		},
	})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(`{"role":"assistant","content":null,"tool_calls":[
		{"id":"a","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"slow\",\"params\":{},\"id\":1}"}},
		{"id":"b","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"get_order\",\"params\":{\"order_id\":\"O\"},\"id\":2}"}}]}`)

	done := make(chan *agent.RunResult)
	go func() {
		res, err := a.Chat(ctx, "go")
		if err != nil {
			t.Error(err)
		}
		done <- res
	}()
	<-started
	if st, _ := a.Status(ctx); st != agent.StatusRunning {
		t.Fatal(st)
	}
	if _, err := e.newAgent(t, a.SessionID()).Chat(ctx, "parallel"); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
	// Stop through a different Agent instance, as a separate HTTP request would.
	if err := e.newAgent(t, a.SessionID()).Stop(ctx); err != nil {
		t.Fatal(err)
	}
	var res *agent.RunResult
	select {
	case res = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("stop did not interrupt the run")
	}
	if res.StopReason != agent.StopStopped || res.Status != agent.StatusIdle {
		t.Fatalf("%+v", res)
	}
	all, _ := a.LatestMessages(ctx, 100)
	if roles(all) != "user,assistant,tool,tool" {
		t.Fatal(roles(all))
	}
	if !strings.Contains(all[3].Content, `"code":-32002`) {
		t.Fatalf("second call should be cancelled: %s", all[3].Content)
	}

	e.llm.push(text("resumed"))
	res, err := a.Continue(ctx)
	if err != nil || res.Reply() != "resumed" {
		t.Fatal(res, err)
	}
}

func TestLLMErrorAndContextLength(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	e.cfg.MaxRetries = -1
	a := e.newAgent(t, "")
	_, err := a.Chat(ctx, "hi") // no scripted reply -> 400
	var apiErr *agent.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("want APIError, got %v", err)
	}
	s, _ := a.Session(ctx)
	if s.Status != agent.StatusIdle || s.LastError == "" {
		t.Fatalf("%+v", s)
	}
	e.llm.push(text("hello"))
	if res, err := a.Continue(ctx); err != nil || res.Reply() != "hello" {
		t.Fatal(res, err)
	}

	e.cfg.ContextLength = 50
	small := e.newAgent(t, a.SessionID())
	if _, err := small.Chat(ctx, strings.Repeat("x", 500)); !errors.Is(err, agent.ErrContextLengthExceeded) {
		t.Fatalf("want ErrContextLengthExceeded, got %v", err)
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
