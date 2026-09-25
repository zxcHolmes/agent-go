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
	"sync/atomic"
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
	send(map[string]any{"choices": []any{map[string]any{"index": 0, "delta": map[string]any{}, "finish_reason": finish}}})
	fmt.Fprint(w, `data: {"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100,"prompt_tokens_details":{"cached_tokens":800},"completion_tokens_details":{"reasoning_tokens":20}}}`+"\n\ndata: [DONE]\n\n")
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
	store agent.Store
	llm   *fakeLLM
	cfg   agent.Config
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
	ctx := context.Background()
	if err := agent.Init(ctx, store); err != nil {
		t.Fatal(err)
	}
	if err := agent.Init(ctx, store); err != nil { // idempotent
		t.Fatal(err)
	}
	f := &fakeLLM{hung: make(chan struct{}, 4)}
	srv := httptest.NewServer(http.HandlerFunc(f.handler))
	t.Cleanup(srv.Close)
	methods = append(methods, agent.NewMethod("get_order", func(ctx context.Context, c *agent.Call, p orderParams) (any, error) {
		return map[string]any{"order_id": p.OrderID, "user": c.Value("user_id")}, nil
	}, agent.MethodDoc{Description: "Get an order"}))
	return &env{store: store, llm: f, cfg: agent.Config{
		BaseURL: srv.URL + "/v1", APIKey: "k", Model: "fake", ContextLength: 100000, MaxOutputTokens: 1000,
		SystemPrompt: "sys", RPCDoc: "docs", ContextParams: map[string]any{"user_id": "u1"},
		Store: store, Billing: agent.Pricing{Input: 100, Output: 1000, CacheRead: 10}, Methods: methods,
		StaleAfter: time.Second, MaxRetries: -1,
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
	if got := statuses(res.Messages); got != "done,done,done,done,done,done" {
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
	for i := range e.llm.requests {
		msgs := e.llm.request(i)
		for j := range prev {
			if !bytes.Equal(prev[j], msgs[j]) {
				t.Fatalf("request %d message %d changed:\n%s\n%s", i, j, prev[j], msgs[j])
			}
		}
		prev = msgs
	}
	if !bytes.Contains(e.llm.requests[0]["stream"], []byte("true")) || !bytes.Contains(e.llm.requests[0]["tools"], []byte(`"name":"json_rpc"`)) {
		t.Fatal("request must stream and carry the tool")
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

	// Cursor pagination.
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

	// Resume the session with a fresh agent.
	e.llm.push(text("again"))
	res, err = e.newAgent(t, sid).Chat(ctx, "thanks")
	if err != nil || res.Reply() != "again" {
		t.Fatal(res, err)
	}
	if n := len(e.llm.request(3)); n != 8 { // system + 6 history + new user
		t.Fatalf("resumed request has %d messages", n)
	}
}

func TestStreamingFlush(t *testing.T) {
	e := setup(t)
	e.cfg.StreamFlushInterval = 150 * time.Millisecond
	var writes, deltas atomic.Int32
	e.cfg.OnMessage = func(ctx context.Context, m agent.Message) {
		if m.Role == "assistant" {
			writes.Add(1)
		}
	}
	e.cfg.OnStream = func(ctx context.Context, id, content, reasoning string) { deltas.Add(1) }
	ctx := context.Background()
	a := e.newAgent(t, "")
	answer := strings.Repeat("流式输出", 6) // 24 runes -> 8 chunks
	e.llm.pushReply(reply{msg: text(answer), delay: 60 * time.Millisecond, hangAfter: -1})

	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "hi")
		done <- err
	}()

	// Poll like a frontend would: cursor = the user message, fetch everything after it.
	var user agent.Message
	waitFor(t, "user message", func() bool {
		ms, _ := a.LatestMessages(ctx, 1)
		if len(ms) == 1 {
			user = ms[0]
		}
		return len(ms) == 1
	})
	var seen []string
	waitFor(t, "stream to finish", func() bool {
		ms, _ := a.MessagesAfter(ctx, user.ID, 10)
		if len(ms) == 0 {
			return false
		}
		m := ms[0]
		if len(seen) == 0 || seen[len(seen)-1] != m.Content {
			seen = append(seen, m.Content)
		}
		if m.Status == agent.MessageStreaming && !strings.HasPrefix(answer, m.Content) {
			t.Fatalf("partial content %q is not a prefix", m.Content)
		}
		return m.Status == agent.MessageDone
	})
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if seen[len(seen)-1] != answer || len(seen) < 3 {
		t.Fatalf("expected several growing snapshots, got %q", seen)
	}
	// 8 deltas over ~480ms with a 150ms interval: far fewer writes than deltas.
	if deltas.Load() != 8 || writes.Load() < 3 || writes.Load() > 6 {
		t.Fatalf("deltas=%d writes=%d", deltas.Load(), writes.Load())
	}
}

func TestConfirmation(t *testing.T) {
	var refunded []string
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		refunded = append(refunded, p.OrderID)
		return "ok", nil
	}, agent.MethodDoc{RequireConfirm: true}))
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
	// The tool message exists already, marked pending.
	if got := statuses(res.Messages); got != "done,done,pending" {
		t.Fatal(got)
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
	all, _ := a.LatestMessages(ctx, 10)
	if got := statuses(all); got != "done,done,done,done" || !strings.Contains(all[2].Content, `"result":"ok"`) {
		t.Fatalf("%s %s", got, all[2].Content)
	}

	// Rejection goes back to the model as an error.
	e.llm.push(toolCall("c2", "refund", `{"order_id":"O-2"}`), text("OK, not refunding."))
	res, _ = a.Chat(ctx, "refund O-2")
	if _, err = a.Confirm(ctx, agent.Reject(res.PendingCalls[0].ID, "too large")); err != nil || len(refunded) != 1 {
		t.Fatal(err, refunded)
	}
	all, _ = a.LatestMessages(ctx, 2)
	if c := all[0].Content; !strings.Contains(c, `"code":-32001`) || !strings.Contains(c, "too large") {
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
	ms, _ := a.LatestMessages(ctx, 10)
	if got := statuses(ms); got != "done,done,running,pending" {
		t.Fatalf("while running: %s", got)
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
	if roles(all) != "user,assistant,tool,tool" || statuses(all) != "done,done,done,done" {
		t.Fatal(roles(all), statuses(all))
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

func TestStopDuringStream(t *testing.T) {
	e := setup(t)
	e.cfg.StreamFlushInterval = time.Hour // only the first delta and the final write
	var deltas atomic.Int32
	e.cfg.OnStream = func(ctx context.Context, id, content, reasoning string) { deltas.Add(1) }
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("Hello world, this is long"), hangAfter: 2})
	done := make(chan *agent.RunResult)
	go func() {
		res, err := a.Chat(ctx, "hi")
		if err != nil {
			t.Error(err)
		}
		done <- res
	}()
	<-e.llm.hung
	waitFor(t, "two deltas", func() bool { return deltas.Load() == 2 })
	if err := a.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	res := <-done
	if res.StopReason != agent.StopStopped || res.Status != agent.StatusIdle {
		t.Fatalf("%+v", res)
	}
	last := res.Messages[len(res.Messages)-1]
	if last.Status != agent.MessageInterrupted || last.Content != "Hello " {
		t.Fatalf("partial message: %+v", last)
	}
	// The partial answer stays in the history for the next turn.
	e.llm.push(text("ok"))
	if _, err := a.Chat(ctx, "go on"); err != nil {
		t.Fatal(err)
	}
	msgs := e.llm.request(1)
	if !strings.Contains(string(msgs[2]), `"content":"Hello "`) {
		t.Fatalf("history: %s", msgs[2])
	}
}

// simulateRestart runs Init on the same database, as a restarted process would.
func simulateRestart(t *testing.T, e *env) {
	t.Helper()
	if err := agent.Init(context.Background(), e.store); err != nil {
		t.Fatal(err)
	}
}

func TestCrashDuringToolCall(t *testing.T) {
	started := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			<-ctx.Done() // the "crashed" run never finishes on its own
			return nil, ctx.Err()
		},
	})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("a", "slow", `{}`))
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "go")
		done <- err
	}()
	<-started

	simulateRestart(t, e)
	s, _ := a.Session(ctx)
	if s.Status != agent.StatusIdle || !strings.Contains(s.LastError, "recovered") {
		t.Fatalf("session after recovery: %+v", s)
	}
	all, _ := a.LatestMessages(ctx, 10)
	if statuses(all) != "done,done,done" || !strings.Contains(all[2].Content, `"code":-32003`) {
		t.Fatalf("%s %s", statuses(all), all[2].Content)
	}
	// The zombie run notices it lost the session and does not overwrite anything.
	if err := <-done; !errors.Is(err, agent.ErrLockLost) {
		t.Fatalf("want ErrLockLost, got %v", err)
	}
	all2, _ := a.LatestMessages(ctx, 10)
	if all2[2].Content != all[2].Content {
		t.Fatal("zombie run overwrote the recovered result")
	}

	e.llm.push(text("That call failed, sorry."))
	res, err := a.Chat(ctx, "what happened?")
	if err != nil || res.Reply() != "That call failed, sorry." {
		t.Fatal(res, err)
	}
}

func TestCrashDuringStream(t *testing.T) {
	e := setup(t)
	e.cfg.StreamFlushInterval = time.Millisecond
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("partial answer here"), hangAfter: 2})
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "hi")
		done <- err
	}()
	<-e.llm.hung
	waitFor(t, "partial flush", func() bool {
		ms, _ := a.LatestMessages(ctx, 1)
		return len(ms) == 1 && ms[0].Content == "partia"
	})
	if st, _ := a.Status(ctx); st != agent.StatusRunning {
		t.Fatal(st)
	}

	simulateRestart(t, e)
	if st, _ := a.Status(ctx); st != agent.StatusIdle {
		t.Fatalf("status after restart: %s", st)
	}
	ms, _ := a.LatestMessages(ctx, 1)
	if ms[0].Status != agent.MessageInterrupted || ms[0].Content != "partia" {
		t.Fatalf("%+v", ms[0])
	}
	if err := <-done; !errors.Is(err, agent.ErrLockLost) {
		t.Fatalf("want ErrLockLost, got %v", err)
	}

	e.llm.push(text("continuing"))
	if res, err := a.Chat(ctx, "go on"); err != nil || res.Reply() != "continuing" {
		t.Fatal(res, err)
	}
}

func TestRecoverStaleOnly(t *testing.T) {
	e := setup(t)
	e.cfg.StaleAfter = 10 * time.Second // heartbeat every ~2.5s
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("abcdef"), hangAfter: 1})
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "hi")
		done <- err
	}()
	<-e.llm.hung
	// A live run on "another instance" must survive a restart that only recovers stale sessions.
	if err := agent.Init(ctx, e.store, agent.RecoverStaleAfter(time.Minute)); err != nil {
		t.Fatal(err)
	}
	if st, _ := a.Status(ctx); st != agent.StatusRunning {
		t.Fatalf("live run was reset: %s", st)
	}
	_ = a.Stop(ctx)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

func TestLLMErrorAndContextLength(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
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
