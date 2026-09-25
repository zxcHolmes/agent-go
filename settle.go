package agent

import (
	"context"
	"fmt"
	"strings"
)

// Bill is a batch of LLM calls settled together.
type Bill struct {
	ID        string        `json:"id"` // use it as the idempotency key in your ledger
	SessionID string        `json:"session_id"`
	Records   []UsageRecord `json:"records"`
	Summary   UsageSummary  `json:"summary"`
}

func summarize(records []UsageRecord) UsageSummary {
	var s UsageSummary
	for _, r := range records {
		s.Calls++
		s.Usage.add(r.Usage)
		s.Cost.add(r.Cost)
	}
	return s
}

// UnbilledUsage returns the LLM calls not marked billed yet, oldest first, and
// their total. sessionID "" means all sessions. Records claimed by a
// SettleUsage that has not completed (BillID set, BilledAt nil) are included.
func UnbilledUsage(ctx context.Context, store Store, sessionID string) (*UsageSummary, []UsageRecord, error) {
	where, args := "billed_at = 0", []any{}
	if sessionID != "" {
		where += " AND session_id = ?"
		args = append(args, sessionID)
	}
	records, err := queryUsage(ctx, store, where, args...)
	if err != nil {
		return nil, nil, err
	}
	s := summarize(records)
	return &s, records, nil
}

// MarkBilled marks LLM call records as billed under billID (empty generates
// one) and returns the bill id. Records already billed are left unchanged, so
// calling it twice is harmless.
func MarkBilled(ctx context.Context, store Store, billID string, recordIDs ...string) (string, error) {
	if billID == "" {
		billID = newID("bill")
	}
	if len(recordIDs) == 0 {
		return billID, nil
	}
	now := nowMillis()
	args := []any{billID, now, now}
	for _, id := range recordIDs {
		args = append(args, id)
	}
	ph := strings.TrimSuffix(strings.Repeat("?, ", len(recordIDs)), ", ")
	_, err := store.Exec(ctx, "UPDATE agent_llm_calls SET bill_id = ?, claimed_at = ?, billed_at = ? WHERE billed_at = 0 AND id IN ("+ph+")", args...)
	return billID, err
}

// SettleUsage bills every unbilled LLM call of a session in one step, safely
// against concurrent settlements:
//
//  1. it atomically claims the session's unclaimed, unbilled records under a
//     new bill id (a record can only be claimed once);
//  2. it calls charge with the bill (e.g. deduct Bill.Summary.Cost.Total from
//     the user's balance, keyed by Bill.ID);
//  3. if charge succeeds the records are marked billed; if it fails they are
//     released for the next attempt and the error is returned.
//
// It returns (nil, nil) without calling charge when there is nothing to bill.
// If the process dies during charge, the records stay claimed (BillID set,
// BilledAt nil, visible in UnbilledUsage): check your ledger for Bill.ID, then
// call CompleteBill or ReleaseBill.
func SettleUsage(ctx context.Context, store Store, sessionID string, charge func(ctx context.Context, bill *Bill) error) (*Bill, error) {
	if sessionID == "" {
		return nil, fmt.Errorf("agent: SettleUsage needs a session id")
	}
	billID := newID("bill")
	if _, err := store.Exec(ctx, "UPDATE agent_llm_calls SET bill_id = ?, claimed_at = ? WHERE session_id = ? AND bill_id = '' AND billed_at = 0",
		billID, nowMillis(), sessionID); err != nil {
		return nil, err
	}
	records, err := queryUsage(ctx, store, "bill_id = ?", billID)
	if err != nil {
		_ = ReleaseBill(context.WithoutCancel(ctx), store, billID)
		return nil, err
	}
	if len(records) == 0 {
		return nil, nil
	}
	bill := &Bill{ID: billID, SessionID: sessionID, Records: records, Summary: summarize(records)}
	if err := charge(ctx, bill); err != nil {
		if rerr := ReleaseBill(context.WithoutCancel(ctx), store, billID); rerr != nil {
			return nil, fmt.Errorf("%w (and releasing bill %s failed: %v)", err, billID, rerr)
		}
		return nil, err
	}
	if err := CompleteBill(context.WithoutCancel(ctx), store, billID); err != nil {
		return bill, fmt.Errorf("agent: charged bill %s but marking it billed failed: %w", billID, err)
	}
	return bill, nil
}

// CompleteBill marks all records claimed under billID as billed.
func CompleteBill(ctx context.Context, store Store, billID string) error {
	_, err := store.Exec(ctx, "UPDATE agent_llm_calls SET billed_at = ? WHERE bill_id = ? AND billed_at = 0", nowMillis(), billID)
	return err
}

// ReleaseBill returns records claimed under billID but not billed to the
// unbilled pool, so the next SettleUsage picks them up again.
func ReleaseBill(ctx context.Context, store Store, billID string) error {
	_, err := store.Exec(ctx, "UPDATE agent_llm_calls SET bill_id = '', claimed_at = 0 WHERE bill_id = ? AND billed_at = 0", billID)
	return err
}
