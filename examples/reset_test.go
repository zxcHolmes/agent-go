package examples

import (
	"context"
	"errors"
	"strings"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
)

func TestResetContext(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(text("A"), text("B"), text("C"), text("D"))
	for _, p := range []string{"a", "b"} {
		if _, err := a.Chat(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	if err := e.client.ResetContext(ctx, a.SessionID()); err != nil {
		t.Fatal(err)
	}
	if _, err := a.Chat(ctx, "c"); err != nil {
		t.Fatal(err)
	}
	// Only what came after the reset reaches the model.
	if got := strings.Join(requestShape(e.llm.request(2)), " | "); got != "system:… | user:c" {
		t.Fatalf("request after reset: %s", got)
	}
	if _, err := a.Chat(ctx, "d"); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(requestShape(e.llm.request(3)), " | "); got != "system:… | user:c | assistant:C | user:d" {
		t.Fatalf("second request after reset: %s", got)
	}
	// Nothing was deleted; the marker is stored for the UI.
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	if len(all) != 9 {
		t.Fatalf("%d messages, want 9", len(all))
	}
	marker := all[4]
	if marker.Kind != agent.MessageKindCompaction || marker.Status != agent.MessageExcluded || marker.RefSeq != marker.Seq+1 {
		t.Fatalf("marker %+v", marker)
	}
	s, _ := e.client.Session(ctx, a.SessionID())
	if s.Status != agent.StatusIdle {
		t.Fatalf("status %s after reset", s.Status)
	}
}

func TestResetContextRefusesAnActiveSession(t *testing.T) {
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		return "ok", nil
	}, agent.MethodDoc{RequireConfirm: true}))
	ctx := context.Background()
	a := e.newAgent(t, "")

	// Waiting for confirmation: the pending call must survive, so no reset.
	e.llm.push(toolCall("c1", "refund", `{"order_id":"O-1"}`))
	if _, err := a.Chat(ctx, "refund O-1"); err != nil {
		t.Fatal(err)
	}
	if err := a.ResetContext(ctx); !errors.Is(err, agent.ErrWaitingConfirmation) {
		t.Fatalf("want ErrWaitingConfirmation, got %v", err)
	}
	if pending, _ := a.PendingCalls(ctx); len(pending) != 1 {
		t.Fatalf("pending calls after refused reset: %d", len(pending))
	}

	// Running: the reset must not land in the middle of a turn.
	b := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("abcdefghi"), hangAfter: 1})
	done := make(chan struct{})
	go func() {
		_, _ = b.Chat(ctx, "hi")
		close(done)
	}()
	<-e.llm.hung
	if err := e.client.ResetContext(ctx, b.SessionID()); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("want ErrBusy, got %v", err)
	}
	if err := b.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	<-done
}

func TestLatestUsage(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	if rec, err := e.client.LatestUsage(ctx, a.SessionID()); err != nil || rec != nil {
		t.Fatalf("new session: %+v, %v", rec, err)
	}
	e.llm.pushReply(reply{msg: text("A"), hangAfter: -1, prompt: 1000})
	e.llm.pushReply(reply{msg: text("B"), hangAfter: -1, prompt: 2500})
	for _, p := range []string{"a", "b"} {
		if _, err := a.Chat(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	rec, err := e.client.LatestUsage(ctx, a.SessionID())
	if err != nil || rec == nil {
		t.Fatalf("latest usage: %+v, %v", rec, err)
	}
	// The newest call, not a sum: 2500 in, 800 of it cached, 100 out.
	if rec.Usage.PromptTokens != 2500 || rec.Usage.CachedTokens != 800 || rec.Usage.CompletionTokens != 100 {
		t.Fatalf("usage %+v", rec.Usage)
	}
	sum, _ := e.client.Usage(ctx, a.SessionID())
	if sum.Usage.PromptTokens != 3500 {
		t.Fatalf("lifetime prompt tokens %d, want 3500", sum.Usage.PromptTokens)
	}
}
