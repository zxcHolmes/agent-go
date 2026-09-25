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

// userContentPart is one part of a multimodal user message.
type userContentPart struct {
	Type     string `json:"type"`
	Text     string `json:"text"`
	ImageURL *struct {
		URL string `json:"url"`
	} `json:"image_url"`
}

// checkUserMessage validates a caller-built user message: role "user" and
// content that is a string or a list of "text" and "image_url" parts. Images
// must be absolute http(s) URLs; inline data (base64 "data:" URLs) is refused.
func checkUserMessage(raw json.RawMessage) error {
	var m struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &m); err != nil {
		return fmt.Errorf("%w: not a JSON object", ErrInvalidUserMessage)
	}
	if m.Role != "user" {
		return fmt.Errorf("%w: role must be \"user\"", ErrInvalidUserMessage)
	}
	var s string
	if json.Unmarshal(m.Content, &s) == nil {
		return nil
	}
	var parts []userContentPart
	if err := json.Unmarshal(m.Content, &parts); err != nil || len(parts) == 0 {
		return fmt.Errorf("%w: content must be a string or a non-empty list of parts", ErrInvalidUserMessage)
	}
	for i, p := range parts {
		switch p.Type {
		case "text":
		case "image_url":
			if p.ImageURL == nil {
				return fmt.Errorf("%w: part %d: image_url is required", ErrInvalidUserMessage, i)
			}
			if err := checkImageURL(p.ImageURL.URL); err != nil {
				return fmt.Errorf("%w: part %d: %v", ErrInvalidUserMessage, i, err)
			}
		default:
			return fmt.Errorf("%w: part %d: unsupported type %q (only text and image_url)", ErrInvalidUserMessage, i, p.Type)
		}
	}
	return nil
}

func enqueue(ctx context.Context, store Store, sessionID string, raw json.RawMessage) (string, error) {
	if err := checkUserMessage(raw); err != nil {
		return "", err
	}
	if _, err := getSession(ctx, store, sessionID); err != nil {
		return "", err
	}
	id := newID("q")
	_, err := store.Exec(ctx, "INSERT INTO agent_queued_messages (id, session_id, content, raw, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
		id, sessionID, decodeMessage(raw).Content, string(raw), queueWaiting, nowMillis())
	if err != nil {
		return "", fmt.Errorf("agent: enqueue: %w", err)
	}
	return id, nil
}

// Queue row states. A running agent claims waiting rows ("taken") before it
// adds them to the conversation, so a cancel racing it either removes the row
// first or finds it taken; never both.
const (
	queueWaiting = "queued"
	queueTaken   = "taken"
)

// queuedMessages lists the rows in the given state, oldest first.
func queuedMessages(ctx context.Context, store Store, sessionID, status string) ([]QueuedMessage, error) {
	rows, err := store.Query(ctx, "SELECT id, session_id, content, raw, created_at FROM agent_queued_messages WHERE session_id = ? AND status = ? ORDER BY created_at ASC, id ASC", sessionID, status)
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

// cancelQueued removes a waiting message. It reads back instead of trusting
// RowsAffected: a row that is still there was taken by a run, and a message
// with the derived id is already in the conversation.
func cancelQueued(ctx context.Context, store Store, sessionID, id string) error {
	if _, err := store.Exec(ctx, "DELETE FROM agent_queued_messages WHERE session_id = ? AND id = ? AND status = ?", sessionID, id, queueWaiting); err != nil {
		return err
	}
	rows, err := store.Query(ctx, "SELECT id FROM agent_queued_messages WHERE session_id = ? AND id = ?", sessionID, id)
	if err != nil {
		return err
	}
	taken := rows.Next()
	err = errors.Join(rows.Err(), rows.Close())
	if err != nil {
		return err
	}
	if taken {
		return ErrQueuedMessageSent
	}
	if _, err := getMessage(ctx, store, sessionID, "msg_"+id); err == nil {
		return ErrQueuedMessageSent
	} else if !errors.Is(err, ErrMessageNotFound) {
		return err
	}
	return nil
}

// deleteQueued drops a row once its message is in the conversation.
func deleteQueued(ctx context.Context, store Store, sessionID, id string) error {
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

// drainQueue moves queued messages into the conversation. It first claims
// the waiting rows (so CancelQueued can no longer remove them), then turns
// each claimed row into a message with a deterministic id before deleting
// it, so a crash in between never loses or duplicates one: rows left taken
// by a crashed run are picked up here by the next one.
func (a *Agent) drainQueue(ctx context.Context, st *runState) (int, error) {
	if _, err := a.store.Exec(st.db, "UPDATE agent_queued_messages SET status = ? WHERE session_id = ? AND status = ?",
		queueTaken, a.sessionID, queueWaiting); err != nil {
		return 0, err
	}
	qs, err := queuedMessages(st.db, a.store, a.sessionID, queueTaken)
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
		if err := deleteQueued(st.db, a.store, a.sessionID, q.ID); err != nil {
			return 0, err
		}
	}
	a.log.Debug("queued messages added to the conversation", "session", a.sessionID, "count", len(qs))
	return len(qs), nil
}

func (a *Agent) hasQueued(st *runState) (bool, error) {
	qs, err := queuedMessages(st.db, a.store, a.sessionID, queueWaiting)
	return len(qs) > 0, err
}
