package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"
)

// QueuedMessage is a user message waiting to join a conversation. Messages
// queued while the agent is running are added right before its next LLM call
// (one role=user message each, in order); messages queued while the session
// is idle are added at the start of the next Chat / Continue.
type QueuedMessage struct {
	ID        string          `json:"id"`
	SessionID string          `json:"session_id"`
	Content   string          `json:"content"`
	Raw       json.RawMessage `json:"raw"`
	CreatedAt time.Time       `json:"created_at"`
}

func userRaw(prompt string) (json.RawMessage, error) {
	return marshalJSON(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"user", prompt})
}

func enqueue(ctx context.Context, store Store, sessionID string, raw json.RawMessage) (string, error) {
	if !json.Valid(raw) {
		return "", errors.New("agent: queued message is not valid JSON")
	}
	if _, err := getSession(ctx, store, sessionID); err != nil {
		return "", err
	}
	id := newID("q")
	_, err := store.Exec(ctx, "INSERT INTO agent_queued_messages (id, session_id, content, raw, created_at) VALUES (?, ?, ?, ?, ?)",
		id, sessionID, decodeMessage(raw).Content, string(raw), nowMillis())
	if err != nil {
		return "", fmt.Errorf("agent: enqueue: %w", err)
	}
	return id, nil
}

func queuedMessages(ctx context.Context, store Store, sessionID string) ([]QueuedMessage, error) {
	rows, err := store.Query(ctx, "SELECT id, session_id, content, raw, created_at FROM agent_queued_messages WHERE session_id = ? ORDER BY created_at ASC, id ASC", sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []QueuedMessage
	for rows.Next() {
		var q QueuedMessage
		var raw string
		var created int64
		if err := rows.Scan(&q.ID, &q.SessionID, &q.Content, &raw, &created); err != nil {
			return nil, err
		}
		q.Raw, q.CreatedAt = json.RawMessage(raw), fromMillis(created)
		out = append(out, q)
	}
	return out, rows.Err()
}

func cancelQueued(ctx context.Context, store Store, sessionID, id string) error {
	_, err := store.Exec(ctx, "DELETE FROM agent_queued_messages WHERE session_id = ? AND id = ?", sessionID, id)
	return err
}

// userMessage builds a user message from raw, appending Config.Reminder to
// what the model sees while Content keeps the text as written.
func (a *Agent) userMessage(raw json.RawMessage) (Message, error) {
	m := newMessage(a.sessionID, raw, MessageDone)
	if a.cfg.Reminder == "" {
		return m, nil
	}
	reminder := "<reminder>\n" + a.cfg.Reminder + "\n</reminder>"
	var msg struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &msg); err != nil {
		return m, err
	}
	var text string
	if json.Unmarshal(msg.Content, &text) == nil {
		b, err := marshalJSON(text + "\n\n" + reminder)
		if err != nil {
			return m, err
		}
		msg.Content = b
	} else {
		var parts []json.RawMessage
		if err := json.Unmarshal(msg.Content, &parts); err != nil {
			return m, fmt.Errorf("agent: user message content must be a string or an array: %w", err)
		}
		part, _ := marshalJSON(struct {
			Type string `json:"type"`
			Text string `json:"text"`
		}{"text", reminder})
		b, err := marshalJSON(append(parts, part))
		if err != nil {
			return m, err
		}
		msg.Content = b
	}
	withReminder, err := marshalJSON(msg)
	if err != nil {
		return m, err
	}
	m.Raw = withReminder // Content stays the user's own text
	return m, nil
}

// drainQueue moves queued messages into the conversation. Each becomes a
// message with a deterministic id before its queue row is deleted, so a crash
// in between never duplicates it.
func (a *Agent) drainQueue(ctx context.Context, st *runState) (int, error) {
	qs, err := queuedMessages(st.db, a.store, a.sessionID)
	if err != nil || len(qs) == 0 {
		return 0, err
	}
	for _, q := range qs {
		id := "msg_" + q.ID
		if _, err := getMessage(st.db, a.store, a.sessionID, id); errors.Is(err, ErrMessageNotFound) {
			m, err := a.userMessage(q.Raw)
			if err != nil {
				return 0, err
			}
			m.ID = id
			if err := a.insertMessage(ctx, st, &m); err != nil {
				return 0, err
			}
		} else if err != nil {
			return 0, err
		}
		if err := cancelQueued(st.db, a.store, a.sessionID, q.ID); err != nil {
			return 0, err
		}
	}
	a.log.Debug("queued messages added to the conversation", "session", a.sessionID, "count", len(qs))
	return len(qs), nil
}

func (a *Agent) hasQueued(st *runState) (bool, error) {
	qs, err := queuedMessages(st.db, a.store, a.sessionID)
	return len(qs) > 0, err
}
