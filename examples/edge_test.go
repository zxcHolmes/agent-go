package examples

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

// checkHistory asserts the invariants every run must leave behind: the
// session is idle, every message is final and every tool call has exactly
// one tool message right after its assistant message.
func checkHistory(t *testing.T, e *env, sid string) []agent.Message {
	t.Helper()
	ctx := context.Background()
	ms, err := e.client.LatestMessages(ctx, sid, 1000)
	if err != nil {
		t.Fatal(err)
	}
	for i, m := range ms {
		if !m.Status.Final() {
			t.Fatalf("message %d (%s) not final: %s", i, m.Role, m.Status)
		}
		if m.Role != "assistant" || len(m.ToolCalls) == 0 {
			continue
		}
		for j, tc := range m.ToolCalls {
			k := i + 1 + j
			if k >= len(ms) || ms[k].Role != "tool" || ms[k].ToolCallID != tc.ID {
				t.Fatalf("tool call %s of message %d has no tool message right after it", tc.ID, i)
			}
		}
	}
	if s, _ := e.client.Session(ctx, sid); s.Status != agent.StatusIdle {
		t.Fatalf("session not idle: %s", s.Status)
	}
	return ms
}

// Cancelling the ctx given to Chat (e.g. the HTTP request that started the
// run went away) does not stop the run: it keeps streaming until Stop.
func TestCallerCtxCancelDoesNotStopStream(t *testing.T) {
	e := setup(t)
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("abcdefghi"), hangAfter: 1})
	ctx, cancel := context.WithCancel(context.Background())
	type out struct {
		res *agent.RunResult
		err error
	}
	done := make(chan out)
	go func() {
		res, err := a.Chat(ctx, "hi")
		done <- out{res, err}
	}()
	<-e.llm.hung
	cancel()
	bg := context.Background()
	select {
	case o := <-done:
		t.Fatalf("run ended with the caller ctx: %+v %v", o.res, o.err)
	case <-time.After(300 * time.Millisecond):
	}
	if st, _ := a.Status(bg); st != agent.StatusRunning {
		t.Fatal(st)
	}
	if err := a.Stop(bg); err != nil {
		t.Fatal(err)
	}
	o := <-done
	if o.err != nil || o.res.StopReason != agent.StopStopped {
		t.Fatalf("%+v %v", o.res, o.err)
	}
	checkHistory(t, e, a.SessionID())
}

// A ctx-aware handler is not cancelled by the caller's ctx either: the call
// finishes and the run completes normally.
func TestCallerCtxCancelDoesNotStopTool(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-release:
				return "done", nil
			}
		},
	})
	a := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "slow", `{}`), text("finished"))
	ctx, cancel := context.WithCancel(context.Background())
	type out struct {
		res *agent.RunResult
		err error
	}
	done := make(chan out)
	go func() {
		res, err := a.Chat(ctx, "go")
		done <- out{res, err}
	}()
	<-started
	cancel()
	time.Sleep(100 * time.Millisecond)
	close(release)
	o := <-done
	if o.err != nil || o.res.Reply() != "finished" {
		t.Fatalf("%+v %v", o.res, o.err)
	}
	ms := checkHistory(t, e, a.SessionID())
	if !strings.Contains(ms[2].Content, `"result":"done"`) {
		t.Fatal(ms[2].Content)
	}
}

// A ctx that is already cancelled never starts a run.
func TestCancelledCtxDoesNotStartRun(t *testing.T) {
	e := setup(t)
	a := e.newAgent(t, "")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := a.Chat(ctx, "hi"); !errors.Is(err, context.Canceled) {
		t.Fatalf("want context.Canceled, got %v", err)
	}
	if ms, _ := e.client.LatestMessages(context.Background(), a.SessionID(), 10); len(ms) != 0 {
		t.Fatalf("%d messages stored", len(ms))
	}
}

// An answer cut by max_tokens ends the run normally; FinishReason tells the
// caller, and Continue lets the model go on from the partial text.
func TestTruncatedAnswer(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("The first part"), hangAfter: -1, finish: "length"})
	res, err := a.Chat(ctx, "write a lot")
	if err != nil || res.FinishReason != "length" || res.StopReason != agent.StopCompleted || res.Reply() != "The first part" {
		t.Fatalf("%+v %v", res, err)
	}
	e.llm.push(text(" and the rest."))
	res, err = a.Continue(ctx)
	if err != nil || res.FinishReason != "stop" {
		t.Fatalf("%+v %v", res, err)
	}
	if got := strings.Join(requestShape(e.llm.request(1))[1:], " | "); got != "user:write a lot | assistant:The first part" {
		t.Fatal(got)
	}
	checkHistory(t, e, a.SessionID())
}

// A tool call whose arguments were cut by max_tokens gets a parse error result
// the model can react to; it never runs and never breaks the history.
func TestTruncatedToolCall(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: `{"role":"assistant","content":null,"tool_calls":[{"id":"c1","type":"function","function":{"name":"json_rpc","arguments":"{\"jsonrpc\":\"2.0\",\"method\":\"get_order\",\"par"}}]}`,
		hangAfter: -1, finish: "length"})
	e.llm.push(text("Sorry, retrying later."))
	res, err := a.Chat(ctx, "get order O")
	if err != nil || res.Reply() != "Sorry, retrying later." {
		t.Fatalf("%+v %v", res, err)
	}
	ms := checkHistory(t, e, a.SessionID())
	if tool := ms[2]; tool.Role != "tool" || !strings.Contains(tool.Content, `"code":-32700`) {
		t.Fatalf("%+v", tool)
	}
}

// settleAndDie claims a bill and "dies" inside charge (a panic stands in for
// the process exiting), leaving the records claimed but not billed.
func settleAndDie(t *testing.T, e *env, sid string) string {
	t.Helper()
	var billID string
	func() {
		defer func() { _ = recover() }()
		_, _ = e.client.SettleUsage(context.Background(), sid, func(ctx context.Context, b *agent.Bill) error {
			billID = b.ID
			panic("process died during charge")
		})
	}()
	if billID == "" {
		t.Fatal("charge not called")
	}
	return billID
}

// A process dying during charge leaves the records claimed: they are visible
// in UnbilledUsage, never claimed by another settlement, and resolved by
// CompleteBill (charge went through) or ReleaseBill (it did not).
func TestSettleCrashDuringCharge(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(text("a"))
	if _, err := a.Chat(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	billID := settleAndDie(t, e, a.SessionID())
	_, recs, _ := e.client.UnbilledUsage(ctx, a.SessionID())
	if len(recs) != 1 || recs[0].BillID != billID || recs[0].ClaimedAt == nil || recs[0].BilledAt != nil {
		t.Fatalf("claimed record: %+v", recs)
	}
	// A new usage record arrives; the next settlement bills only that one.
	e.llm.push(text("b"))
	if _, err := a.Chat(ctx, "more"); err != nil {
		t.Fatal(err)
	}
	bill, err := e.client.SettleUsage(ctx, a.SessionID(), func(context.Context, *agent.Bill) error { return nil })
	if err != nil || bill == nil || bill.Summary.Calls != 1 || bill.Records[0].ID == recs[0].ID {
		t.Fatalf("%+v %v", bill, err)
	}
	// The ledger says the charge never happened: release and settle again.
	if err := e.client.ReleaseBill(ctx, billID); err != nil {
		t.Fatal(err)
	}
	bill, err = e.client.SettleUsage(ctx, a.SessionID(), func(context.Context, *agent.Bill) error { return nil })
	if err != nil || bill == nil || bill.Summary.Calls != 1 || bill.Records[0].ID != recs[0].ID {
		t.Fatalf("%+v %v", bill, err)
	}

	// The ledger says the charge went through: complete it; nothing is left.
	e.llm.push(text("c"))
	if _, err := a.Chat(ctx, "again"); err != nil {
		t.Fatal(err)
	}
	billID = settleAndDie(t, e, a.SessionID())
	if err := e.client.CompleteBill(ctx, billID); err != nil {
		t.Fatal(err)
	}
	if sum, _, _ := e.client.UnbilledUsage(ctx, a.SessionID()); sum.Calls != 0 {
		t.Fatalf("still unbilled: %+v", sum)
	}
	if b, err := e.client.SettleUsage(ctx, a.SessionID(), func(context.Context, *agent.Bill) error { t.Error("charged twice"); return nil }); b != nil || err != nil {
		t.Fatal(b, err)
	}
	// Releasing a completed bill must not reopen it.
	if err := e.client.ReleaseBill(ctx, billID); err != nil {
		t.Fatal(err)
	}
	if sum, _, _ := e.client.UnbilledUsage(ctx, a.SessionID()); sum.Calls != 0 {
		t.Fatal("ReleaseBill reopened a billed record")
	}
}

func confirmEnv(t *testing.T, ran *atomic.Int32) (*env, *agent.Agent, string) {
	t.Helper()
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		ran.Add(1)
		return "ok", nil
	}, agent.MethodDoc{RequireConfirm: true}))
	a := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "refund", `{"order_id":"O-1"}`))
	res, err := a.Chat(context.Background(), "refund")
	if err != nil || len(res.PendingCalls) != 1 {
		t.Fatal(res, err)
	}
	return e, a, res.PendingCalls[0].ID
}

// Two requests confirming the same call at once: it runs exactly once.
func TestConfirmConcurrent(t *testing.T) {
	var ran atomic.Int32
	e, a, callID := confirmEnv(t, &ran)
	ctx := context.Background()
	e.llm.push(text("Refunded."))
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	for i := 0; i < 2; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := e.newAgent(t, a.SessionID()).Confirm(ctx, agent.Approve(callID))
			errs <- err
		}()
	}
	wg.Wait()
	close(errs)
	var ok int
	for err := range errs {
		switch {
		case err == nil:
			ok++
		case errors.Is(err, agent.ErrBusy), errors.Is(err, agent.ErrNoPendingCalls), errors.Is(err, agent.ErrCallNotPending):
		default:
			t.Fatal(err)
		}
	}
	if ok != 1 || ran.Load() != 1 {
		t.Fatalf("%d confirms succeeded, handler ran %d times", ok, ran.Load())
	}
	checkHistory(t, e, a.SessionID())
}

// Confirm racing Stop: whoever wins, the session ends idle with a valid
// history and the call ran at most once.
func TestConfirmRacesStop(t *testing.T) {
	for i := 0; i < 20; i++ {
		var ran atomic.Int32
		e, a, callID := confirmEnv(t, &ran)
		ctx := context.Background()
		e.llm.push(text("Refunded."))
		done := make(chan error)
		go func() {
			_, err := a.Confirm(ctx, agent.Approve(callID))
			done <- err
		}()
		time.Sleep(time.Duration(i%4) * time.Millisecond)
		if err := e.client.Stop(ctx, a.SessionID()); err != nil {
			t.Fatal(err)
		}
		if err := <-done; err != nil && !errors.Is(err, agent.ErrNoPendingCalls) && !errors.Is(err, agent.ErrCallNotPending) {
			t.Fatalf("iteration %d: %v", i, err)
		}
		waitFor(t, "idle", func() bool { st, _ := a.Status(ctx); return st == agent.StatusIdle })
		ms := checkHistory(t, e, a.SessionID())
		if ran.Load() > 1 {
			t.Fatalf("iteration %d: ran %d times", i, ran.Load())
		}
		if ran.Load() == 0 && !strings.Contains(ms[2].Content, "stopped by user") {
			t.Fatalf("iteration %d: not run but result is %s", i, ms[2].Content)
		}
	}
}

// A process that dies after approving a call but before running it: the next
// run closes the call as not executed instead of running it unattended.
func TestCrashAfterApproval(t *testing.T) {
	var ran atomic.Int32
	e, a, callID := confirmEnv(t, &ran)
	ctx := context.Background()
	old := time.Now().Add(-time.Hour).UnixMilli()
	if _, err := e.store.Exec(ctx, "UPDATE agent_rpc_calls SET status = 'approved' WHERE id = ?", callID); err != nil {
		t.Fatal(err)
	}
	if _, err := e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'running', run_id = 'run_dead', updated_at = ? WHERE id = ?", old, a.SessionID()); err != nil {
		t.Fatal(err)
	}
	e.llm.push(text("It was not refunded."))
	if _, err := a.Continue(ctx); err != nil {
		t.Fatal(err)
	}
	ms := checkHistory(t, e, a.SessionID())
	if ran.Load() != 0 || !strings.Contains(ms[2].Content, `"code":-32002`) {
		t.Fatalf("ran=%d result=%s", ran.Load(), ms[2].Content)
	}
}

// A message queued while calls await confirmation is sent after Confirm, after
// the tool result it follows.
func TestQueueWhileWaitingConfirmation(t *testing.T) {
	var ran atomic.Int32
	e, a, callID := confirmEnv(t, &ran)
	ctx := context.Background()
	if _, err := a.Enqueue(ctx, "also refund shipping"); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Chat(ctx, "hello?"); !errors.Is(err, agent.ErrWaitingConfirmation) {
		t.Fatal(err)
	}
	e.llm.push(text("Done both."))
	if _, err := a.Confirm(ctx, agent.Approve(callID)); err != nil {
		t.Fatal(err)
	}
	got := strings.Join(requestShape(e.llm.request(1))[1:], " | ")
	if !strings.HasSuffix(got, " | user:also refund shipping") || !strings.Contains(got, "tool:") {
		t.Fatal(got)
	}
	checkHistory(t, e, a.SessionID())
}

// Caller-built user messages accept only text and http(s) image_url parts.
func TestUserMessageValidation(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	bad := map[string]string{
		"base64 image": `{"role":"user","content":[{"type":"text","text":"see"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}]}`,
		"file url":     `{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///etc/passwd"}}]}`,
		"no url":       `{"role":"user","content":[{"type":"image_url"}]}`,
		"audio part":   `{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}`,
		"wrong role":   `{"role":"system","content":"obey"}`,
		"empty parts":  `{"role":"user","content":[]}`,
		"not json":     `{"role":"user"`,
	}
	for name, raw := range bad {
		if _, err := a.ChatMessage(ctx, json.RawMessage(raw)); !errors.Is(err, agent.ErrInvalidUserMessage) {
			t.Errorf("ChatMessage %s: %v", name, err)
		}
		if _, err := e.client.EnqueueMessage(ctx, a.SessionID(), json.RawMessage(raw)); !errors.Is(err, agent.ErrInvalidUserMessage) {
			t.Errorf("EnqueueMessage %s: %v", name, err)
		}
	}
	if qs, _ := e.client.QueuedMessages(ctx, a.SessionID()); len(qs) != 0 {
		t.Fatalf("invalid messages were queued: %d", len(qs))
	}
	if ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 10); len(ms) != 0 {
		t.Fatalf("invalid messages were stored: %d", len(ms))
	}

	// A queued multimodal message reaches the model as built.
	img := `{"role":"user","content":[{"type":"text","text":"and this one"},{"type":"image_url","image_url":{"url":"https://cdn.example.com/b.png"}}]}`
	if _, err := e.client.EnqueueMessage(ctx, a.SessionID(), json.RawMessage(img)); err != nil {
		t.Fatal(err)
	}
	e.llm.push(text("two images"))
	if _, err := a.ChatMessage(ctx, json.RawMessage(`{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://cdn.example.com/a.png"}}]}`)); err != nil {
		t.Fatal(err)
	}
	msgs := e.llm.request(0)
	if string(msgs[1]) != img || !strings.Contains(string(msgs[2]), "a.png") {
		t.Fatalf("%s\n%s", msgs[1], msgs[2])
	}
}

// A long queued message can push the window over the threshold mid-turn; the
// compaction then starts the context at that message and the request stays
// valid (no tool message without its call).
func TestQueuedMessageTriggersCompaction(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{Name: "wait", Handler: func(ctx context.Context, c *agent.Call) (any, error) {
		close(started)
		<-release
		return "ok", nil
	}})
	e.cfg.ContextLength = 3000
	e.cfg.MaxOutputTokens = 100
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: toolCall("c1", "wait", `{}`), hangAfter: -1, prompt: 1500})
	e.llm.push(text("answered"))
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "start")
		done <- err
	}()
	<-started
	if _, err := a.Enqueue(ctx, strings.Repeat("long detail ", 300)); err != nil {
		t.Fatal(err)
	}
	close(release)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	shape := requestShape(e.llm.request(1))
	if len(shape) != 2 || !strings.HasPrefix(shape[1], "user:long detail") {
		t.Fatalf("%v", shape)
	}
	ms := checkHistory(t, e, a.SessionID())
	var compacted bool
	for _, m := range ms {
		compacted = compacted || m.Kind == agent.MessageKindCompaction
	}
	if !compacted {
		t.Fatal("no compaction message")
	}
}

// Request body options reach the provider as configured.
func TestRequestBodyOptions(t *testing.T) {
	e := setup(t)
	e.cfg.UseMaxCompletionTokens = true
	e.cfg.ExtraBody = map[string]any{"temperature": 0.2, "stream_options": nil, "model": "hijack"}
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(text("ok"))
	if _, err := a.Chat(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	e.llm.mu.Lock()
	req := e.llm.requests[0]
	e.llm.mu.Unlock()
	if _, ok := req["max_tokens"]; ok || string(req["max_completion_tokens"]) == "" {
		t.Fatalf("max tokens fields: %s / %s", req["max_tokens"], req["max_completion_tokens"])
	}
	if string(req["temperature"]) != "0.2" || string(req["model"]) != `"fake"` {
		t.Fatalf("temperature %s model %s", req["temperature"], req["model"])
	}
	if _, ok := req["stream_options"]; ok {
		t.Fatal("stream_options not removed")
	}
}

// A stream that goes silent is aborted after StreamIdleTimeout. Before any
// output it is retried; after output it is not (the partial text is kept).
func TestStreamIdleTimeout(t *testing.T) {
	e := setup(t)
	e.cfg.StreamIdleTimeout = 200 * time.Millisecond
	e.cfg.MaxRetries = 1
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("never sent"), hangAfter: 0})
	e.llm.push(text("second try"))
	res, err := a.Chat(ctx, "hi")
	if err != nil || res.Reply() != "second try" {
		t.Fatalf("%+v %v", res, err)
	}

	e.llm.pushReply(reply{msg: text("abcdefghi"), hangAfter: 1})
	start := time.Now()
	_, err = a.Chat(ctx, "again")
	if err == nil || !strings.Contains(err.Error(), "idle timeout") || time.Since(start) > 3*time.Second {
		t.Fatalf("%v after %v", err, time.Since(start))
	}
	ms := checkHistory(t, e, a.SessionID())
	if last := ms[len(ms)-1]; last.Status != agent.MessageInterrupted || last.Content != "abc" {
		t.Fatalf("%+v", last)
	}
}

// DeleteSession refuses a live run; paging with MessagesBefore / MessagesAfter
// walks the whole history once, in order.
func TestDeleteBusyAndPaging(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	for i := 0; i < 7; i++ {
		e.llm.push(text(fmt.Sprintf("reply %d", i)))
		if _, err := a.Chat(ctx, fmt.Sprintf("question %d", i)); err != nil {
			t.Fatal(err)
		}
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	if len(all) != 14 {
		t.Fatal(len(all))
	}
	// Backwards, 3 at a time.
	page, _ := e.client.LatestMessages(ctx, a.SessionID(), 3)
	got := append([]agent.Message(nil), page...)
	for len(page) > 0 {
		page, _ = e.client.MessagesBefore(ctx, a.SessionID(), got[0].ID, 3)
		got = append(append([]agent.Message(nil), page...), got...)
	}
	// Forwards, 4 at a time.
	fwd := []agent.Message{all[0]}
	page, _ = e.client.MessagesAfter(ctx, a.SessionID(), all[0].ID, 4)
	for len(page) > 0 {
		fwd = append(fwd, page...)
		page, _ = e.client.MessagesAfter(ctx, a.SessionID(), fwd[len(fwd)-1].ID, 4)
	}
	for i := range all {
		if got[i].ID != all[i].ID || fwd[i].ID != all[i].ID {
			t.Fatalf("paging order differs at %d", i)
		}
	}
	if len(got) != len(all) || len(fwd) != len(all) {
		t.Fatalf("paged %d / %d of %d", len(got), len(fwd), len(all))
	}

	e.llm.pushReply(reply{msg: text("abcdef"), hangAfter: 1})
	done := make(chan struct{})
	go func() {
		_, _ = a.Chat(ctx, "long")
		close(done)
	}()
	<-e.llm.hung
	if err := e.client.DeleteSession(ctx, a.SessionID()); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
	_ = a.Stop(ctx)
	<-done
	if err := e.client.DeleteSession(ctx, a.SessionID()); err != nil {
		t.Fatal(err)
	}
	if _, err := e.client.Session(ctx, a.SessionID()); !errors.Is(err, agent.ErrSessionNotFound) {
		t.Fatal(err)
	}
}
