package examples

import (
	"context"
	"errors"
	"sync"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

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
