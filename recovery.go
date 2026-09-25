package agent

import (
	"context"
	"fmt"
	"time"
)

// addedColumns lists columns introduced after the first published schema
// (v0.6.1). CREATE TABLE IF NOT EXISTS does not touch existing tables, so
// migrate adds them to databases created by older versions.
var addedColumns = []struct{ table, column, ddl string }{
	{"agent_messages", "ref_seq", "BIGINT NOT NULL DEFAULT 0"}, // v0.7.0: compaction
}

// migrate creates the SDK tables if they do not exist and adds columns that
// older versions did not have.
func migrate(ctx context.Context, store Store) error {
	for _, stmt := range SchemaStatements(store.Dialect()) {
		if _, err := store.Exec(ctx, stmt); err != nil {
			return fmt.Errorf("agent: init schema: %w", err)
		}
	}
	for _, c := range addedColumns {
		rows, err := store.Query(ctx, "SELECT "+c.column+" FROM "+c.table+" WHERE 1 = 0")
		if err == nil {
			rows.Close()
			continue // already there
		}
		if _, err := store.Exec(ctx, "ALTER TABLE "+c.table+" ADD COLUMN "+c.column+" "+c.ddl); err != nil {
			return fmt.Errorf("agent: add column %s.%s: %w", c.table, c.column, err)
		}
	}
	return nil
}

// recoverCrashed resets sessions left "running" or "stopping" by a dead
// process. staleAfter > 0 limits it to sessions without a heartbeat for that
// long.
func recoverCrashed(ctx context.Context, store Store, staleAfter time.Duration) error {
	q := "SELECT id FROM agent_sessions WHERE status IN (?, ?)"
	args := []any{string(StatusRunning), string(StatusStopping)}
	if staleAfter > 0 {
		q += " AND updated_at < ?"
		args = append(args, nowMillis()-staleAfter.Milliseconds())
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
		if err := resetSession(ctx, store, id); err != nil {
			return fmt.Errorf("agent: recover session %s: %w", id, err)
		}
	}
	return nil
}

const crashNote = "recovered: the process stopped while this session was running"

// resetSession recovers a session whose run died with its process:
//   - assistant messages still "streaming" become "interrupted" (partial
//     content is kept and stays in the history);
//   - RPC calls that were running become "failed" with a CodeCrashed error
//     result (they may or may not have taken effect);
//   - queued / approved calls that never started become "cancelled";
//   - calls awaiting confirmation are kept, unless the session was stopping
//     (then they are cancelled as stopped by user);
//   - the session becomes idle, or waiting_confirmation if calls still await
//     approval, and last_error records the recovery.
//
// NewClient runs it for every crashed session.
func resetSession(ctx context.Context, store Store, id string) error {
	s, err := getSession(ctx, store, id)
	if err != nil {
		return err
	}
	if err := recoverSessionData(ctx, store, id); err != nil {
		return err
	}
	pending, err := pendingCalls(ctx, store, id)
	if err != nil {
		return err
	}
	if s.Status == StatusStopping {
		for i := range pending {
			if err := completeCall(ctx, store, &pending[i], CallCancelled, rpcErrorResponse(pending[i].RPCID, errStoppedNotRun)); err != nil {
				return err
			}
		}
		pending = nil
	}
	status := StatusIdle
	if len(pending) > 0 {
		status = StatusWaitingConfirmation
	}
	_, err = store.Exec(ctx, "UPDATE agent_sessions SET status = ?, run_id = '', last_error = ?, updated_at = ? WHERE id = ?",
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
