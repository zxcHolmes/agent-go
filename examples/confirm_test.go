package examples

import (
	"context"
	"errors"
	"strings"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

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
