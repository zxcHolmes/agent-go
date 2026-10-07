package examples

import (
	"context"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
)

func TestActivity(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			<-release
			return "ok", nil
		},
	})
	ctx := context.Background()

	idle := e.newAgent(t, "")
	e.llm.push(text("hi"))
	if _, err := idle.Chat(ctx, "hello"); err != nil {
		t.Fatal(err)
	}

	busy := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "slow", `{}`), text("done"))
	done := make(chan struct{})
	go func() {
		defer close(done)
		if _, err := busy.Chat(ctx, "go"); err != nil {
			t.Error(err)
		}
	}()
	<-started
	if _, err := e.client.Enqueue(ctx, busy.SessionID(), "also this"); err != nil {
		t.Fatal(err)
	}

	act, err := e.client.Activity(ctx, agent.ActivityOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if act.Running != 1 || act.Stale != 0 || act.WaitingConfirmation != 0 || act.QueuedMessages != 1 {
		t.Fatalf("while running: %+v", act)
	}
	if len(act.Sessions) != 2 || act.Sessions[0].ID != busy.SessionID() || act.Sessions[0].Status != agent.StatusRunning || act.Sessions[0].Stale {
		t.Fatalf("running session must come first and be live: %+v", act.Sessions)
	}
	if s := act.Sessions[1]; s.ID != idle.SessionID() || s.LastMessageAt.IsZero() || s.LastMessageAt.Before(s.CreatedAt) {
		t.Fatalf("idle session: %+v", s)
	}

	close(release)
	<-done
	waitFor(t, "idle", func() bool { st, _ := busy.Status(ctx); return st == agent.StatusIdle })

	// A session left "running" by a dead process: no heartbeat for longer
	// than StaleAfter (1s in tests).
	if _, err := e.db.Exec("UPDATE agent_sessions SET status = 'running', updated_at = 1 WHERE id = ?", idle.SessionID()); err != nil {
		t.Fatal(err)
	}
	act, err = e.client.Activity(ctx, agent.ActivityOptions{Limit: 1})
	if err != nil {
		t.Fatal(err)
	}
	if act.Running != 0 || act.Stale != 1 || len(act.Sessions) != 1 || !act.Sessions[0].Stale || act.Sessions[0].ID != idle.SessionID() {
		t.Fatalf("stale: %+v", act)
	}
}
