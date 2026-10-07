package agent

import (
	"context"
	"fmt"
	"time"
)

// Activity is a store-wide snapshot of what the agent is doing right now,
// across every session and every process sharing the store. Its main use is
// operational: "is anything running that a restart or deploy would cut off?".
type Activity struct {
	// Running counts sessions that are running or stopping with a live
	// heartbeat: a run some process is executing right now.
	Running int `json:"running"`
	// Stale counts sessions still marked running or stopping whose heartbeat
	// is older than Config.StaleAfter: their process died. Nothing is
	// executing them; the next run, NewClient recovery or ResetSession
	// clears them.
	Stale int `json:"stale"`
	// WaitingConfirmation counts sessions paused on calls awaiting a decision.
	// No run is active for them.
	WaitingConfirmation int `json:"waiting_confirmation"`
	// QueuedMessages counts user messages not yet added to a conversation
	// (waiting, or claimed by a run that has not written them yet).
	QueuedMessages int `json:"queued_messages"`
	// Total counts every session in the store, for paging Sessions.
	Total int `json:"total"`
	// Sessions is one page of sessions: those not idle first, then the most
	// recently updated (ActivityOptions.Limit / Offset).
	Sessions []SessionActivity `json:"sessions"`
	// CheckedAt is when the snapshot was taken, by this process's clock.
	CheckedAt time.Time `json:"checked_at"`
}

// SessionActivity is one session in an Activity snapshot.
type SessionActivity struct {
	Session
	// Stale is true for a running or stopping session whose heartbeat is
	// older than Config.StaleAfter (see Activity.Stale).
	Stale bool `json:"stale"`
	// LastMessageAt is when the session's newest message was last written;
	// zero if it has none. UpdatedAt is the heartbeat while a run is active,
	// so this is the better "last activity" for a conversation.
	LastMessageAt time.Time `json:"last_message_at"`
}

// ActivityOptions configures Client.Activity.
type ActivityOptions struct {
	// Limit caps Activity.Sessions (default 50, max 500). The counts always
	// cover every session.
	Limit int
	// Offset skips that many sessions of the ordering, for paging.
	Offset int
}

// Activity reports how many runs are in flight across the store and lists the
// most recent sessions. It is read-only and costs three queries whatever the
// number of sessions.
func (c *Client) Activity(ctx context.Context, opts ActivityOptions) (*Activity, error) {
	limit := opts.Limit
	if limit <= 0 {
		limit = 50
	} else if limit > 500 {
		limit = 500
	}
	offset := opts.Offset
	if offset < 0 {
		offset = 0
	}
	now := time.Now()
	staleBefore := now.Add(-c.cfg.StaleAfter).UnixMilli()
	out := &Activity{CheckedAt: now, Sessions: []SessionActivity{}}

	rows, err := c.store.Query(ctx, `SELECT
		COALESCE(SUM(CASE WHEN status IN (?, ?) AND updated_at >= ? THEN 1 ELSE 0 END), 0) AS running,
		COALESCE(SUM(CASE WHEN status IN (?, ?) AND updated_at < ? THEN 1 ELSE 0 END), 0) AS stale,
		COALESCE(SUM(CASE WHEN status = ? THEN 1 ELSE 0 END), 0) AS waiting,
		COUNT(*) AS sessions
		FROM agent_sessions`,
		string(StatusRunning), string(StatusStopping), staleBefore,
		string(StatusRunning), string(StatusStopping), staleBefore,
		string(StatusWaitingConfirmation))
	if err != nil {
		return nil, fmt.Errorf("agent: activity counts: %w", err)
	}
	var running, stale, waiting, total int64
	if rows.Next() {
		err = rows.Scan(&running, &stale, &waiting, &total)
	}
	if err == nil {
		err = rows.Err()
	}
	rows.Close()
	if err != nil {
		return nil, fmt.Errorf("agent: activity counts: %w", err)
	}
	out.Running, out.Stale, out.WaitingConfirmation, out.Total = int(running), int(stale), int(waiting), int(total)

	rows, err = c.store.Query(ctx, "SELECT COUNT(*) AS queued FROM agent_queued_messages WHERE status IN (?, ?)",
		queueWaiting, queueTaken)
	if err != nil {
		return nil, fmt.Errorf("agent: activity queue: %w", err)
	}
	var queued int64
	if rows.Next() {
		err = rows.Scan(&queued)
	}
	if err == nil {
		err = rows.Err()
	}
	rows.Close()
	if err != nil {
		return nil, fmt.Errorf("agent: activity queue: %w", err)
	}
	out.QueuedMessages = int(queued)

	// The newest message is read by seq through the (session_id, seq) unique
	// key, never with MAX over a whole conversation.
	rows, err = c.store.Query(ctx, `SELECT s.id, s.status, s.last_error, s.metadata, s.created_at, s.updated_at,
		COALESCE((SELECT m.updated_at FROM agent_messages m WHERE m.session_id = s.id ORDER BY m.seq DESC LIMIT 1), 0) AS last_message_at
		FROM agent_sessions s
		ORDER BY CASE WHEN s.status = ? THEN 1 ELSE 0 END, s.updated_at DESC, s.id
		LIMIT ? OFFSET ?`, string(StatusIdle), limit, offset)
	if err != nil {
		return nil, fmt.Errorf("agent: activity sessions: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var s SessionActivity
		var status, meta string
		var created, updated, lastMsg int64
		if err := rows.Scan(&s.ID, &status, &s.LastError, &meta, &created, &updated, &lastMsg); err != nil {
			return nil, fmt.Errorf("agent: activity sessions: %w", err)
		}
		s.Status = Status(status)
		s.Metadata = rawOrNil(meta)
		s.CreatedAt, s.UpdatedAt = fromMillis(created), fromMillis(updated)
		if lastMsg > 0 {
			s.LastMessageAt = fromMillis(lastMsg)
		}
		s.Stale = (s.Status == StatusRunning || s.Status == StatusStopping) && updated < staleBefore
		out.Sessions = append(out.Sessions, s)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("agent: activity sessions: %w", err)
	}
	return out, nil
}
