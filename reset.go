package agent

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ResetContext starts the model's context afresh without deleting anything:
// from the next message on, the model sees only what comes after this point,
// while the whole transcript stays in the store for the UI and the session id
// keeps working.
//
// It records a compaction message (Kind "compaction", status excluded) whose
// RefSeq points just past itself, so contextWindow reads nothing older. No
// summary is written — use it when the history is no longer wanted, not when
// it is merely long (automatic compaction handles that).
//
// The session must be idle: a running session returns ErrBusy, one waiting
// for confirmation ErrWaitingConfirmation. The reset holds the session lock
// while it writes, so it can never land in the middle of a run's turn. The
// session's last_error is preserved. After a reset, continue with Chat (or
// Enqueue + Continue); Continue alone would send the model an empty history.
func (a *Agent) ResetContext(ctx context.Context) error {
	s, err := getSession(ctx, a.store, a.sessionID)
	if err != nil {
		return err
	}
	runID, err := a.acquire(ctx, []Status{StatusIdle})
	if errors.Is(err, ErrNoPendingCalls) {
		return ErrBusy // lost a race with a run that has since finished
	}
	if err != nil {
		return err
	}
	st := &runState{runID: runID, lease: lease{a.sessionID, runID}, db: context.WithoutCancel(ctx)}
	// The session may have been taken over from a run that died (acquire
	// accepts a stale lock). Repair its tail first, exactly as a run would, so
	// the history left behind the reset point is consistent.
	status := StatusIdle
	resetErr := a.heal(st.db, st)
	if resetErr == nil {
		// A dead run can leave calls awaiting confirmation; resetting past them
		// would strand them, so hand the session back in that state instead.
		pending, perr := pendingCalls(st.db, a.store, a.sessionID)
		switch {
		case perr != nil:
			resetErr = perr
		case len(pending) > 0:
			status, resetErr = StatusWaitingConfirmation, ErrWaitingConfirmation
		default:
			resetErr = a.insertResetMarker(st)
		}
	}
	if rerr := a.release(st, status, s.LastError); rerr != nil {
		resetErr = errors.Join(resetErr, rerr)
	}
	if resetErr == nil {
		a.log.Info("context reset", "session", a.sessionID)
	}
	return resetErr
}

// insertResetMarker writes the compaction message. Its seq is only known once
// inserted, so RefSeq is set in a second statement; a crash between the two
// leaves RefSeq 0, which reads the whole history — the reset simply did not
// happen, and nothing is lost.
func (a *Agent) insertResetMarker(st *runState) error {
	note := "Context reset: earlier messages are no longer sent to the model."
	raw, err := marshalJSON(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"system", note})
	if err != nil {
		return err
	}
	now := time.Now()
	m := Message{
		ID: newID("msg"), SessionID: a.sessionID, Role: "system", Kind: MessageKindCompaction,
		Status: MessageExcluded, Content: note, Raw: raw, CreatedAt: now, UpdatedAt: now,
	}
	if err := insertMessage(st.db, a.store, &m); err != nil {
		return err
	}
	q, args := st.lease.fence("UPDATE agent_messages SET ref_seq = ? WHERE id = ?", []any{m.Seq + 1, m.ID})
	if _, err := a.store.Exec(st.db, q, args...); err != nil {
		return fmt.Errorf("agent: set reset point: %w", err)
	}
	return nil
}

// ResetContext resets a session's context without building an agent; see
// Agent.ResetContext.
func (c *Client) ResetContext(ctx context.Context, sessionID string) error {
	cfg := c.cfg.withDefaults()
	a := &Agent{client: c, cfg: cfg, store: c.store, sessionID: sessionID, methods: map[string]Method{}, res: c.res, log: c.log}
	for _, m := range cfg.Methods {
		a.methods[m.Name] = m
	}
	return a.ResetContext(ctx)
}
