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

// SendResult tells how Send delivered a message.
type SendResult struct {
	// QueueID is the id the message had in the queue.
	QueueID string
	// Run is the result of the run Send started itself because the session
	// was idle; nil when the message went to another run.
	Run *RunResult
	// Queued is true when the message was handed to a run that was already
	// active (it is answered by that run), or left in the queue behind calls
	// awaiting confirmation (it is sent once they are confirmed).
	Queued bool
}

// Send delivers a user message whatever the session is doing, so callers do
// not have to check the status first:
//
//   - idle: it starts a run like Chat and blocks until the run ends;
//   - running / stopping: the message joins the active run's next LLM call;
//     Send returns as soon as that run has taken it. If the run ends first
//     (the message arrived just as it finished), Send starts a new run itself,
//     so the message is always answered;
//   - waiting for confirmation: the message stays queued and is sent after
//     Confirm; Send returns immediately.
//
// The message is stored in the queue before anything else, so it survives a
// crash. If ctx ends while waiting, Send returns ctx's error and the message
// stays queued for the next run.
func (a *Agent) Send(ctx context.Context, prompt string) (*SendResult, error) {
	id, err := a.Enqueue(ctx, prompt)
	if err != nil {
		return nil, err
	}
	out := &SendResult{QueueID: id}
	for {
		res, err := a.run(ctx, []Status{StatusIdle}, func(ctx context.Context, st *runState) error {
			n, err := a.prepareQueued(ctx, st)
			if err == nil && n == 0 {
				return errNothingQueued // another run already answered it
			}
			return err
		}, true)
		switch {
		case err == nil:
			out.Run = res
			return out, nil
		case errors.Is(err, ErrWaitingConfirmation), errors.Is(err, errNothingQueued):
			out.Queued = true
			return out, nil
		case !errors.Is(err, ErrBusy):
			return out, err
		}
		// Another run is active: wait until it takes the message or ends.
		for {
			if still, err := a.isQueued(ctx, id); err != nil {
				return out, err
			} else if !still {
				out.Queued = true
				return out, nil
			}
			s, err := getSession(ctx, a.store, a.sessionID)
			if err != nil {
				return out, err
			}
			if s.Status != StatusRunning && s.Status != StatusStopping {
				break // it ended without our message: start a run ourselves
			}
			t := time.NewTimer(100 * time.Millisecond)
			select {
			case <-ctx.Done():
				t.Stop()
				return out, context.Cause(ctx)
			case <-t.C:
			}
		}
	}
}

// prepareQueuedRun is Chat's preparation without a new prompt: close calls
// left by an interrupted run, then add the queued messages.
func (a *Agent) prepareQueuedRun(ctx context.Context, st *runState) error {
	_, err := a.prepareQueued(ctx, st)
	return err
}

// errNothingQueued ends a Send run that found its message already taken.
var errNothingQueued = errors.New("agent: nothing queued")

func (a *Agent) prepareQueued(ctx context.Context, st *runState) (int, error) {
	open, err := openCalls(st.db, a.store, a.sessionID)
	if err != nil {
		return 0, err
	}
	for _, c := range open {
		if c.Status == CallAwaitingConfirmation {
			return 0, ErrWaitingConfirmation
		}
	}
	if err := a.cancelCalls(ctx, st, open, false, &RPCError{Code: CodeCancelled, Message: "cancelled: superseded by a new user message"}); err != nil {
		return 0, err
	}
	return a.drainQueue(ctx, st)
}

func (a *Agent) isQueued(ctx context.Context, id string) (bool, error) {
	rows, err := a.store.Query(ctx, "SELECT id FROM agent_queued_messages WHERE session_id = ? AND id = ?", a.sessionID, id)
	if err != nil {
		return false, err
	}
	defer rows.Close()
	return rows.Next(), rows.Err()
}
