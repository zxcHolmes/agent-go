package agent

import (
	"context"
	"fmt"
	"time"
)

// InitOption configures Init.
type InitOption func(*initOptions)

type initOptions struct {
	staleAfter time.Duration
	noRecovery bool
}

// RecoverStaleAfter limits crash recovery to sessions whose run has not sent a
// heartbeat for d. Use it when several processes share one database, so a
// restarting instance does not reset runs that are alive on other instances
// (runs heartbeat every few seconds; see Config.StaleAfter).
func RecoverStaleAfter(d time.Duration) InitOption {
	return func(o *initOptions) { o.staleAfter = d }
}

// WithoutRecovery makes Init only create tables.
func WithoutRecovery() InitOption {
	return func(o *initOptions) { o.noRecovery = true }
}

// Init creates the SDK tables if they do not exist and recovers sessions left
// "running" by a previous process (crash, kill, deploy). Call it once at
// startup; it is idempotent.
//
// By default every running session is considered dead, which is right when a
// single process uses the database. With several processes, pass
// RecoverStaleAfter. See ResetSession for what recovery does.
func Init(ctx context.Context, store Store, opts ...InitOption) error {
	var o initOptions
	for _, fn := range opts {
		fn(&o)
	}
	for _, stmt := range SchemaStatements(store.Dialect()) {
		if _, err := store.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("agent: init schema: %w", err)
		}
	}
	if o.noRecovery {
		return nil
	}
	q := "SELECT id FROM agent_sessions WHERE status = ?"
	args := []any{string(StatusRunning)}
	if o.staleAfter > 0 {
		q += " AND updated_at < ?"
		args = append(args, nowMillis()-o.staleAfter.Milliseconds())
	}
	rows, err := store.Query(ctx, q, args...)
	if err != nil {
		return err
	}
	var ids []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			rows.Close()
			return err
		}
		ids = append(ids, id)
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return err
	}
	for _, id := range ids {
		if err := ResetSession(ctx, store, id); err != nil {
			return fmt.Errorf("agent: recover session %s: %w", id, err)
		}
	}
	return nil
}

const crashNote = "recovered: the process stopped while this session was running"

// ResetSession recovers a session whose run died with its process:
//   - assistant messages still "streaming" become "interrupted" (partial
//     content is kept and stays in the history);
//   - RPC calls that were running become "failed" with a CodeCrashed error
//     result (they may or may not have taken effect);
//   - queued / approved calls that never started become "cancelled";
//   - calls awaiting confirmation are kept;
//   - the session becomes idle, or waiting_confirmation if calls still await
//     approval, and last_error records the recovery.
//
// Init calls it for every crashed session. Only call it directly when you know
// no run is active for the session.
func ResetSession(ctx context.Context, store Store, id string) error {
	if _, err := GetSession(ctx, store, id); err != nil {
		return err
	}
	if err := recoverSessionData(ctx, store, id); err != nil {
		return err
	}
	pending, err := PendingCalls(ctx, store, id)
	if err != nil {
		return err
	}
	status := StatusIdle
	if len(pending) > 0 {
		status = StatusWaitingConfirmation
	}
	_, err = store.Exec(ctx, "UPDATE agent_sessions SET status = ?, run_id = '', stop_requested = 0, last_error = ?, updated_at = ? WHERE id = ?",
		string(status), crashNote, nowMillis(), id)
	return err
}

// recoverSessionData fixes messages and calls left mid-flight by a dead run.
func recoverSessionData(ctx context.Context, store Store, sessionID string) error {
	if _, err := store.Exec(ctx, "UPDATE agent_messages SET status = ?, updated_at = ? WHERE session_id = ? AND status = ?",
		string(MessageInterrupted), nowMillis(), sessionID, string(MessageStreaming)); err != nil {
		return err
	}
	open, err := openCalls(ctx, store, sessionID)
	if err != nil {
		return err
	}
	for i := range open {
		c := &open[i]
		switch c.Status {
		case CallRunning:
			err = completeCall(ctx, store, c, CallFailed, rpcErrorResponse(c.RPCID, &RPCError{
				Code:    CodeCrashed,
				Message: "failed: the system crashed while this call was running; it may or may not have taken effect",
			}))
		case CallQueued, CallApproved:
			err = completeCall(ctx, store, c, CallCancelled, rpcErrorResponse(c.RPCID, &RPCError{
				Code:    CodeCancelled,
				Message: "cancelled: the system restarted before this call ran",
			}))
		}
		if err != nil {
			return err
		}
	}
	return nil
}
