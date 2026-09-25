package examples

import (
	"context"
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
	e.cfg.ToolConcurrency = 1 // this test covers the call that has not started yet
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
	if time.Since(start) > 100*time.Millisecond {
		t.Fatalf("Stop took %v", time.Since(start))
	}
	// Stop does not wait: the run finishes stopping on its own.
	waitFor(t, "idle", func() bool { st, _ := a.Status(ctx); return st == agent.StatusIdle })
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

// Stop of a run owned by another process only marks the session; it does not
// wait for that process to notice.
func TestStopDoesNotWaitForRemoteRun(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	if _, err := e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'running', run_id = 'run_remote', updated_at = ? WHERE id = ?",
		time.Now().UnixMilli(), a.SessionID()); err != nil {
		t.Fatal(err)
	}
	start := time.Now()
	if err := a.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > 100*time.Millisecond {
		t.Fatalf("Stop took %v", time.Since(start))
	}
	if st, _ := a.Status(ctx); st != agent.StatusStopping {
		t.Fatal(st)
	}
	if err := a.Stop(ctx); err != nil { // repeated while stopping
		t.Fatal(err)
	}
	if _, err := a.Chat(ctx, "hi"); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
	// The remote process died before finishing the stop: the next run takes over once it is stale.
	old := time.Now().Add(-time.Hour).UnixMilli()
	if _, err := e.store.Exec(ctx, "UPDATE agent_sessions SET updated_at = ? WHERE id = ?", old, a.SessionID()); err != nil {
		t.Fatal(err)
	}
	e.llm.push(text("back"))
	if res, err := a.Chat(ctx, "hi"); err != nil || res.Reply() != "back" {
		t.Fatal(res, err)
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
	if _, err := e.store.Exec(ctx, `INSERT INTO agent_messages (id, session_id, seq, role, kind, ref_seq, status, content, reasoning, tool_call_id, raw, created_at, updated_at)
		VALUES ('msg_dead', ?, 1, 'assistant', '', 0, 'streaming', 'half', '', '', '{"role":"assistant","content":"half"}', ?, ?)`, a.SessionID(), old, old); err != nil {
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
