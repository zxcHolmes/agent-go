package agent

import (
	"context"
	"fmt"
)

// createSession creates a new idle session and returns its id. metadata is
// optional and stored as JSON (e.g. the owning user id).
func createSession(ctx context.Context, store Store, metadata map[string]any) (string, error) {
	meta := []byte("{}")
	if metadata != nil {
		var err error
		if meta, err = marshalJSON(metadata); err != nil {
			return "", fmt.Errorf("agent: encode metadata: %w", err)
		}
	}
	id := newID("ses")
	now := nowMillis()
	_, err := store.Exec(ctx,
		"INSERT INTO agent_sessions (id, status, run_id, last_error, metadata, created_at, updated_at) VALUES (?, ?, '', '', ?, ?, ?)",
		id, string(StatusIdle), string(meta), now, now)
	if err != nil {
		return "", fmt.Errorf("agent: create session: %w", err)
	}
	return id, nil
}

// getSession loads a session.
func getSession(ctx context.Context, store Store, id string) (*Session, error) {
	rows, err := store.Query(ctx, "SELECT id, status, last_error, metadata, created_at, updated_at FROM agent_sessions WHERE id = ?", id)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	if !rows.Next() {
		if err := rows.Err(); err != nil {
			return nil, err
		}
		return nil, ErrSessionNotFound
	}
	var s Session
	var status, meta string
	var created, updated int64
	if err := rows.Scan(&s.ID, &status, &s.LastError, &meta, &created, &updated); err != nil {
		return nil, err
	}
	s.Status = Status(status)
	s.Metadata = rawOrNil(meta)
	s.CreatedAt, s.UpdatedAt = fromMillis(created), fromMillis(updated)
	return &s, nil
}
