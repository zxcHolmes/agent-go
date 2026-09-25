package examples

import (
	"context"
	"strings"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
)

func TestSendIdleStartsRun(t *testing.T) {
	e := setup(t)
	a := e.newAgent(t, "")
	e.llm.push(text("hello back"))
	r, err := a.Send(context.Background(), "hello")
	if err != nil || r.Run == nil || r.Queued || r.Run.Reply() != "hello back" {
		t.Fatalf("%+v %v", r, err)
	}
	if got := requestShape(e.llm.request(0)); got[len(got)-1] != "user:hello" {
		t.Fatal(got)
	}
}

func TestSendWhileRunningJoinsRun(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{Name: "slow", Handler: func(ctx context.Context, c *agent.Call) (any, error) {
		close(started)
		<-release
		return "done", nil
	}})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "slow", `{}`), text("answered both"))
	done := make(chan struct{})
	go func() { _, _ = a.Chat(ctx, "task"); close(done) }()
	<-started
	sent := make(chan *agent.SendResult)
	go func() {
		r, err := e.newAgent(t, a.SessionID()).Send(ctx, "one more detail")
		if err != nil {
			t.Error(err)
		}
		sent <- r
	}()
	time.Sleep(100 * time.Millisecond)
	close(release)
	r := <-sent
	<-done
	if !r.Queued || r.Run != nil || len(e.llm.requests) != 2 {
		t.Fatalf("%+v, %d requests", r, len(e.llm.requests))
	}
	if got := requestShape(e.llm.request(1)); got[len(got)-1] != "user:one more detail" {
		t.Fatal(got)
	}
}

// The gap: a run that is about to finish (already past its last queue check)
// cannot take the message; Send must answer it with a run of its own.
func TestSendAtEndOfRunIsNotLost(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	// Simulate a live run on another instance in its final moments.
	_, _ = e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'running', run_id = 'run_other', updated_at = ? WHERE id = ?", time.Now().UnixMilli(), a.SessionID())
	e.llm.push(text("got your message"))
	sent := make(chan *agent.SendResult)
	go func() {
		r, err := a.Send(ctx, "late message")
		if err != nil {
			t.Error(err)
		}
		sent <- r
	}()
	time.Sleep(300 * time.Millisecond)
	if q, _ := e.client.QueuedMessages(ctx, a.SessionID()); len(q) != 1 {
		t.Fatalf("message must wait in the queue while the other run is active: %d", len(q))
	}
	// The other run finishes without having seen the message.
	_, _ = e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'idle', run_id = '' WHERE id = ?", a.SessionID())
	r := <-sent
	if r.Run == nil || r.Run.Reply() != "got your message" {
		t.Fatalf("%+v", r)
	}
}

func TestSendWhileWaitingConfirmation(t *testing.T) {
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		return "ok", nil
	}, agent.MethodDoc{RequireConfirm: true}))
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "refund", `{"order_id":"O-1"}`), text("refunded and noted"))
	res, _ := a.Chat(ctx, "refund")
	r, err := a.Send(ctx, "please also email me")
	if err != nil || !r.Queued || r.Run != nil {
		t.Fatalf("%+v %v", r, err)
	}
	if _, err := a.Confirm(ctx, agent.Approve(res.PendingCalls[0].ID)); err != nil {
		t.Fatal(err)
	}
	got := requestShape(e.llm.request(1))
	if !strings.HasPrefix(got[len(got)-2], "tool:") || got[len(got)-1] != "user:please also email me" {
		t.Fatal(got)
	}
}
