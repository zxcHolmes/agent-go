package agent

import "context"

// acquire atomically moves the session to running if it is in one of the
// allowed states, or if a previous run stopped heartbeating (crash).
func (a *Agent) acquire(ctx context.Context, from []Status) (string, error) {
	runID := newID("run")
	now := nowMillis()
	args := []any{string(StatusRunning), runID, now, a.sessionID}
	for _, s := range from {
		args = append(args, string(s))
	}
	args = append(args, string(StatusRunning), string(StatusStopping), now-a.cfg.StaleAfter.Milliseconds())
	ph := placeholders(len(from))
	if _, err := a.store.Exec(ctx,
		"UPDATE agent_sessions SET status = ?, run_id = ?, last_error = '', updated_at = ? WHERE id = ? AND (status IN ("+ph+") OR (status IN (?, ?) AND updated_at < ?))",
		args...); err != nil {
		return "", err
	}
	// Verify by reading back: rows-affected is unreliable across drivers.
	owner, _, err := a.sessionLock(ctx)
	if err != nil {
		return "", err
	}
	if owner == runID {
		return runID, nil
	}
	s, err := getSession(ctx, a.store, a.sessionID)
	if err != nil {
		return "", err
	}
	switch s.Status {
	case StatusRunning, StatusStopping:
		return "", ErrBusy
	case StatusWaitingConfirmation:
		return "", ErrWaitingConfirmation
	default:
		return "", ErrNoPendingCalls
	}
}

func (a *Agent) sessionLock(ctx context.Context) (runID string, status Status, err error) {
	rows, err := a.store.Query(ctx, "SELECT run_id, status FROM agent_sessions WHERE id = ?", a.sessionID)
	if err != nil {
		return "", "", err
	}
	defer rows.Close()
	if !rows.Next() {
		if err := rows.Err(); err != nil {
			return "", "", err
		}
		return "", "", ErrSessionNotFound
	}
	var st string
	err = rows.Scan(&runID, &st)
	return runID, Status(st), err
}

// checkpoint heartbeats the lock and honours stop requests.
func (a *Agent) checkpoint(ctx context.Context, st *runState) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	if _, err := a.store.Exec(st.db, "UPDATE agent_sessions SET updated_at = ? WHERE id = ? AND run_id = ?", nowMillis(), a.sessionID, st.runID); err != nil {
		return err
	}
	owner, status, err := a.sessionLock(st.db)
	if err != nil {
		return err
	}
	if owner != st.runID {
		return ErrLockLost
	}
	if status == StatusStopping {
		st.cancel(ErrStopped)
		return ErrStopped
	}
	return nil
}

// ownsLock reports whether this run still holds the session, e.g. after a
// long RPC call during which Init may have recovered the session.
func (a *Agent) ownsLock(st *runState) error {
	owner, _, err := a.sessionLock(st.db)
	if err != nil {
		return err
	}
	if owner != st.runID {
		return ErrLockLost
	}
	return nil
}

// lease fences the writes of a run: an UPDATE carrying it only applies while
// runID still owns the session. Checking ownership first (ownsLock) leaves a
// window in which the session can be taken over (crash recovery, a stale-run
// takeover) before the write lands; the fence closes it, so a run that lost
// the lock never overwrites what the new owner wrote, e.g. a final tool
// message's raw bytes. The zero lease is unfenced (recovery outside a run).
type lease struct{ sessionID, runID string }

// fence appends the ownership condition to a query ending in a WHERE clause.
func (l lease) fence(query string, args []any) (string, []any) {
	if l.runID == "" {
		return query, args
	}
	return query + " AND EXISTS (SELECT 1 FROM agent_sessions WHERE id = ? AND run_id = ?)", append(args, l.sessionID, l.runID)
}

func (a *Agent) release(st *runState, status Status, lastErr string) error {
	_, err := a.store.Exec(st.db, "UPDATE agent_sessions SET status = ?, run_id = '', last_error = ?, updated_at = ? WHERE id = ? AND run_id = ?",
		string(status), lastErr, nowMillis(), a.sessionID, st.runID)
	return err
}
