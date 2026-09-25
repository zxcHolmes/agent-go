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
	return &env{store: store, client: client, llm: f, cfg: cfg}
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

func TestChatToolLoop(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	sid, err := e.client.CreateSession(ctx, map[string]any{"owner": "u1"})
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
	sum, err := e.client.Usage(ctx, a.SessionID())
	if err != nil {
		t.Fatal(err)
	}
	wantCredits := 3 * (200*100.0/1e6 + 800*10.0/1e6 + 100*1000.0/1e6)
	if sum.Calls != 3 || sum.Usage.CachedTokens != 2400 || sum.Usage.ReasoningTokens != 60 || abs(sum.Cost.Total-wantCredits) > 1e-9 {
		t.Fatalf("usage %+v", sum)
	}

	// Cursor pagination.
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	latest, _ := e.client.LatestMessages(ctx, a.SessionID(), 2)
	if len(all) != 6 || roles(latest) != "tool,assistant" || latest[1].ID != all[5].ID {
		t.Fatalf("latest %s", roles(latest))
	}
	before, _ := e.client.MessagesBefore(ctx, a.SessionID(), all[3].ID, 2)
	after, _ := e.client.MessagesAfter(ctx, a.SessionID(), all[3].ID, 10)
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
		ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 5)
		for _, m := range ms {
			if m.Role == "user" {
				user = m
				return true
			}
		}
		return false
	})
	var seen []string
	waitFor(t, "stream to finish", func() bool {
		ms, _ := e.client.MessagesAfter(ctx, a.SessionID(), user.ID, 10)
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
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if got := statuses(all); got != "done,done,done,done" || !strings.Contains(all[2].Content, `"result":"ok"`) {
		t.Fatalf("%s %s", got, all[2].Content)
	}

	// Rejection goes back to the model as an error.
	e.llm.push(toolCall("c2", "refund", `{"order_id":"O-2"}`), text("OK, not refunding."))
	res, _ = a.Chat(ctx, "refund O-2")
	if _, err = a.Confirm(ctx, agent.Reject(res.PendingCalls[0].ID, "too large")); err != nil || len(refunded) != 1 {
		t.Fatal(err, refunded)
	}
	all, _ = e.client.LatestMessages(ctx, a.SessionID(), 2)
	if c := all[0].Content; !strings.Contains(c, `"code":-32001`) || !strings.Contains(c, "too large") {
		t.Fatalf("rejection: %s", c)
	}
	if _, err := a.Confirm(ctx, agent.Approve("x")); !errors.Is(err, agent.ErrNoPendingCalls) {
		t.Fatal(err)
	}
}

func TestStopAndContinue(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			<-release // ignores ctx on purpose: Stop must not wait for it
			return "late result", nil
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
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if got := statuses(ms); got != "done,done,running,pending" {
		t.Fatalf("while running: %s", got)
	}
	if _, err := e.newAgent(t, a.SessionID()).Chat(ctx, "parallel"); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
	// Stop through a different Agent instance, as a separate HTTP request would.
	start := time.Now()
	if err := e.newAgent(t, a.SessionID()).Stop(ctx); err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > time.Second {
		t.Fatalf("Stop took %v", time.Since(start))
	}
	// Stop is synchronous: the session is idle as soon as it returns.
	if st, _ := a.Status(ctx); st != agent.StatusIdle {
		t.Fatal(st)
	}
	res := <-done
	if res.StopReason != agent.StopStopped || res.Status != agent.StatusIdle {
		t.Fatalf("%+v", res)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	if roles(all) != "user,assistant,tool,tool" || statuses(all) != "done,done,done,done" {
		t.Fatal(roles(all), statuses(all))
	}
	if !strings.Contains(all[2].Content, "stopped by user while this call was running") ||
		!strings.Contains(all[3].Content, "stopped by user: this call was not executed") {
		t.Fatalf("results:\n%s\n%s", all[2].Content, all[3].Content)
	}
	// The abandoned handler finishing later changes nothing.
	close(release)
	time.Sleep(50 * time.Millisecond)
	after, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	if after[2].Content != all[2].Content {
		t.Fatal("late handler result overwrote the stop result")
	}
	// Repeated Stop on an idle session is a no-op.
	if err := a.Stop(ctx); err != nil {
		t.Fatal(err)
	}

	e.llm.push(text("resumed"))
	res, err := a.Chat(ctx, "continue") // no ErrBusy right after Stop
	if err != nil || res.Reply() != "resumed" {
		t.Fatal(res, err)
	}
}

func TestStopWhileWaitingConfirmation(t *testing.T) {
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		t.Error("must not run")
		return "", nil
	}, agent.MethodDoc{RequireConfirm: true}))
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "refund", `{"order_id":"O-1"}`))
	if res, err := a.Chat(ctx, "refund"); err != nil || res.Status != agent.StatusWaitingConfirmation {
		t.Fatal(res, err)
	}
	if err := e.client.Stop(ctx, a.SessionID()); err != nil { // no Agent needed, e.g. a "stop" HTTP handler
		t.Fatal(err)
	}
	s, _ := a.Session(ctx)
	p, _ := a.PendingCalls(ctx)
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 1)
	if s.Status != agent.StatusIdle || len(p) != 0 || !strings.Contains(ms[0].Content, "stopped by user") || ms[0].Status != agent.MessageDone {
		t.Fatalf("%+v %v %+v", s, p, ms[0])
	}
}

func TestStopConcurrentAndRepeated(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("abcdefghi"), hangAfter: 1})
	done := make(chan *agent.RunResult)
	go func() {
		res, _ := a.Chat(ctx, "hi")
		done <- res
	}()
	<-e.llm.hung
	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs <- e.newAgent(t, a.SessionID()).Stop(ctx)
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		if err != nil {
			t.Fatal(err)
		}
	}
	if res := <-done; res.StopReason != agent.StopStopped {
		t.Fatalf("%+v", res)
	}
	if st, _ := a.Status(ctx); st != agent.StatusIdle {
		t.Fatal(st)
	}
}

// Stop racing a run that ends on its own must always leave the session idle.
func TestStopRacesCompletion(t *testing.T) {
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		return "ok", nil
	}, agent.MethodDoc{RequireConfirm: true}))
	ctx := context.Background()
	a := e.newAgent(t, "")
	for i := 0; i < 30; i++ {
		if i%2 == 0 {
			e.llm.push(text("done"))
		} else {
			e.llm.push(toolCall(fmt.Sprintf("c%d", i), "refund", `{"order_id":"O"}`)) // ends waiting for confirmation
		}
		done := make(chan error)
		go func() {
			_, err := a.Chat(ctx, "hi")
			done <- err
		}()
		time.Sleep(time.Duration(i%5) * time.Millisecond)
		if err := a.Stop(ctx); err != nil {
			t.Fatal(err)
		}
		if err := <-done; err != nil && !errors.Is(err, agent.ErrBusy) {
			t.Fatalf("iteration %d: %v", i, err)
		}
		if err := a.Stop(ctx); err != nil { // Chat may have started after the first Stop
			t.Fatal(err)
		}
		s, _ := a.Session(ctx)
		p, _ := a.PendingCalls(ctx)
		if s.Status != agent.StatusIdle || len(p) != 0 {
			t.Fatalf("iteration %d: status %s, %d pending", i, s.Status, len(p))
		}
		e.llm.mu.Lock()
		e.llm.replies = nil // drop the reply if Stop won before the request
		e.llm.mu.Unlock()
	}
}

func TestStopFromAnotherProcess(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("abcdefghi"), hangAfter: 1})
	done := make(chan *agent.RunResult)
	go func() {
		res, _ := a.Chat(ctx, "hi")
		done <- res
	}()
	<-e.llm.hung
	// What Stop in another process writes; the local run has no in-memory signal.
	if _, err := e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'stopping' WHERE id = ?", a.SessionID()); err != nil {
		t.Fatal(err)
	}
	select {
	case res := <-done:
		if res.StopReason != agent.StopStopped || res.Status != agent.StatusIdle {
			t.Fatalf("%+v", res)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("remote stop not noticed")
	}
}

func TestStopDeadRun(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	// A run whose process died mid-stream, long ago.
	old := time.Now().Add(-time.Hour).UnixMilli()
	if _, err := e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'running', run_id = 'run_dead', updated_at = ? WHERE id = ?", old, a.SessionID()); err != nil {
		t.Fatal(err)
	}
	if _, err := e.store.Exec(ctx, `INSERT INTO agent_messages (id, session_id, seq, role, kind, status, content, reasoning, tool_call_id, raw, created_at, updated_at)
		VALUES ('msg_dead', ?, 1, 'assistant', '', 'streaming', 'half', '', '', '{"role":"assistant","content":"half"}', ?, ?)`, a.SessionID(), old, old); err != nil {
		t.Fatal(err)
	}
	if err := a.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	m, _ := e.client.Message(ctx, a.SessionID(), "msg_dead")
	if st, _ := a.Status(ctx); st != agent.StatusIdle || m.Status != agent.MessageInterrupted {
		t.Fatalf("%s %s", st, m.Status)
	}
}

func TestBillingSettlement(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "get_order", `{"order_id":"O"}`), text("a"))
	if _, err := a.Chat(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	sum, recs, err := e.client.UnbilledUsage(ctx, a.SessionID())
	perCall := 200*100.0/1e6 + 800*10.0/1e6 + 100*1000.0/1e6
	if err != nil || sum.Calls != 2 || len(recs) != 2 || abs(sum.Cost.Total-2*perCall) > 1e-9 {
		t.Fatalf("%+v %v", sum, err)
	}

	// Charge fails: records go back to the pool.
	if _, err := e.client.SettleUsage(ctx, a.SessionID(), func(ctx context.Context, b *agent.Bill) error { return errors.New("no balance") }); err == nil {
		t.Fatal("want error")
	}
	_, recs, _ = e.client.UnbilledUsage(ctx, a.SessionID())
	if len(recs) != 2 || recs[0].BillID != "" {
		t.Fatalf("not released: %+v", recs)
	}

	var charged float64
	bill, err := e.client.SettleUsage(ctx, a.SessionID(), func(ctx context.Context, b *agent.Bill) error {
		charged += b.Summary.Cost.Total
		return nil
	})
	if err != nil || bill == nil || bill.Summary.Calls != 2 || abs(charged-2*perCall) > 1e-9 {
		t.Fatalf("%+v %v", bill, err)
	}
	if sum, _, _ := e.client.UnbilledUsage(ctx, a.SessionID()); sum.Calls != 0 {
		t.Fatal("still unbilled")
	}
	if b, err := e.client.SettleUsage(ctx, a.SessionID(), func(context.Context, *agent.Bill) error { t.Error("nothing to charge"); return nil }); b != nil || err != nil {
		t.Fatal(b, err)
	}
	all, _ := e.client.ListUsage(ctx, a.SessionID())
	if all[0].BillID != bill.ID || all[0].BilledAt == nil {
		t.Fatalf("%+v", all[0])
	}

	// Concurrent settlements never charge a record twice.
	for i := 0; i < 5; i++ {
		e.llm.push(text("x"))
		if _, err := a.Chat(ctx, "more"); err != nil {
			t.Fatal(err)
		}
	}
	var mu sync.Mutex
	var billedCalls int64
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := e.client.SettleUsage(ctx, a.SessionID(), func(ctx context.Context, b *agent.Bill) error {
				mu.Lock()
				billedCalls += b.Summary.Calls
				mu.Unlock()
				return nil
			})
			if err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if billedCalls != 5 {
		t.Fatalf("charged %d calls, want 5", billedCalls)
	}

	// Manual marking.
	e.llm.push(text("y"))
	_, _ = a.Chat(ctx, "again")
	_, recs, _ = e.client.UnbilledUsage(ctx, "") // all sessions
	if len(recs) != 1 {
		t.Fatal(len(recs))
	}
	if _, err := e.client.MarkBilled(ctx, "ledger-42", recs[0].ID); err != nil {
		t.Fatal(err)
	}
	if sum, _, _ := e.client.UnbilledUsage(ctx, ""); sum.Calls != 0 {
		t.Fatal("MarkBilled did not mark")
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

// simulateRestart creates a new client on the same database, as a restarted process would.
func simulateRestart(t *testing.T, e *env) {
	t.Helper()
	if _, err := agent.NewClient(context.Background(), e.cfg); err != nil {
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
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if statuses(all) != "done,done,done" || !strings.Contains(all[2].Content, `"code":-32003`) {
		t.Fatalf("%s %s", statuses(all), all[2].Content)
	}
	// The zombie run notices it lost the session and does not overwrite anything.
	if err := <-done; !errors.Is(err, agent.ErrLockLost) {
		t.Fatalf("want ErrLockLost, got %v", err)
	}
	all2, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
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
		ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 1)
		return len(ms) == 1 && ms[0].Content == "partia"
	})
	if st, _ := a.Status(ctx); st != agent.StatusRunning {
		t.Fatal(st)
	}

	simulateRestart(t, e)
	if st, _ := a.Status(ctx); st != agent.StatusIdle {
		t.Fatalf("status after restart: %s", st)
	}
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 1)
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
	cfg := e.cfg
	cfg.RecoverStaleAfter = time.Minute
	if _, err := agent.NewClient(ctx, cfg); err != nil {
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

func TestClientOverridesAndReadOnly(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	client, err := agent.NewClient(ctx, e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, err := client.Agent(ctx, "", agent.AgentOptions{Override: func(c *agent.Config) {
		c.SystemPrompt = "you are a pirate"
		c.Methods = nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	e.llm.push(text("arr"))
	if _, err := a.Chat(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	msgs := e.llm.request(0)
	if !strings.Contains(string(msgs[0]), "you are a pirate") || strings.Contains(string(msgs[0]), "get_order") {
		t.Fatalf("override not applied: %s", msgs[0])
	}
	if _, ok := e.llm.requests[0]["tools"]; ok {
		t.Fatal("no methods -> no tools")
	}

	// A client without LLM settings still serves reads and billing.
	ro, err := agent.NewClient(ctx, agent.Config{Store: e.store})
	if err != nil {
		t.Fatal(err)
	}
	if ms, err := ro.LatestMessages(ctx, a.SessionID(), 5); err != nil || len(ms) != 2 {
		t.Fatal(ms, err)
	}
	if sum, _, err := ro.UnbilledUsage(ctx, a.SessionID()); err != nil || sum.Calls != 1 {
		t.Fatal(sum, err)
	}
	if _, err := ro.Agent(ctx, "", agent.AgentOptions{}); err == nil {
		t.Fatal("agent without BaseURL/Model must fail")
	}
}

func TestViewImage(t *testing.T) {
	e := setup(t)
	e.cfg.ViewImage = true
	e.cfg.ViewImageDetail = "low"
	ctx := context.Background()
	a := e.newAgent(t, "")
	args, _ := json.Marshal(`{"url":"https://cdn.example.com/cat.png"}`)
	bad, _ := json.Marshal(`{"url":"ftp://x/y.png"}`)
	rpcArgs, _ := json.Marshal(`{"jsonrpc":"2.0","method":"get_order","params":{"order_id":"O"},"id":1}`)
	// One turn mixing a JSON-RPC call, a valid and an invalid view_image call.
	e.llm.push(fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[
		{"id":"t1","type":"function","function":{"name":"view_image","arguments":%s}},
		{"id":"t2","type":"function","function":{"name":"json_rpc","arguments":%s}},
		{"id":"t3","type":"function","function":{"name":"view_image","arguments":%s}}]}`, args, rpcArgs, bad),
		text("A cat."))
	res, err := a.Chat(ctx, "what is in this picture?")
	if err != nil || res.Reply() != "A cat." {
		t.Fatal(res, err)
	}
	if got := roles(res.Messages); got != "user,assistant,tool,tool,tool,user,assistant" {
		t.Fatal(got)
	}
	img := res.Messages[5]
	if img.Kind != agent.MessageKindViewImage || img.ToolCallID != "t1" || img.Status != agent.MessageDone {
		t.Fatalf("%+v", img)
	}
	if !strings.Contains(res.Messages[4].Content, `"ok":false`) {
		t.Fatalf("invalid url accepted: %s", res.Messages[4].Content)
	}

	// The tool is offered, and the model's next request carries the image part, unmodified.
	if !bytes.Contains(e.llm.requests[0]["tools"], []byte(`"name":"view_image"`)) {
		t.Fatal("view_image tool not offered")
	}
	msgs := e.llm.request(1)
	last := string(msgs[len(msgs)-1])
	if !strings.Contains(last, `{"type":"image_url","image_url":{"url":"https://cdn.example.com/cat.png","detail":"low"}}`) ||
		!strings.Contains(last, `{"role":"user","content":[{"type":"text"`) {
		t.Fatalf("image message: %s", last)
	}
	if !strings.Contains(string(msgs[len(msgs)-2]), `"tool_call_id":"t3"`) {
		t.Fatal("image must come after every tool result")
	}
}

func abs(f float64) float64 {
	if f < 0 {
		return -f
	}
	return f
}
