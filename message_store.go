package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const messageCols = "id, session_id, seq, role, kind, ref_seq, status, content, reasoning, tool_call_id, raw, created_at, updated_at"

func scanMessages(rows Rows) ([]Message, error) {
	defer rows.Close()
	var out []Message
	for rows.Next() {
		var m Message
		var raw, status string
		var created, updated int64
		if err := rows.Scan(&m.ID, &m.SessionID, &m.Seq, &m.Role, &m.Kind, &m.RefSeq, &status, &m.Content, &m.Reasoning, &m.ToolCallID, &raw, &created, &updated); err != nil {
			return nil, err
		}
		m.Status = MessageStatus(status)
		m.Raw = json.RawMessage(raw)
		m.CreatedAt, m.UpdatedAt = fromMillis(created), fromMillis(updated)
		m.ToolCalls = decodeMessage(m.Raw).ToolCalls
		out = append(out, m)
	}
	return out, rows.Err()
}

type decodedMessage struct {
	Role       string
	Content    string
	ToolCallID string
	ToolCalls  []ToolCall
}

func decodeMessage(raw json.RawMessage) decodedMessage {
	var m struct {
		Role       string          `json:"role"`
		Content    json.RawMessage `json:"content"`
		ToolCallID string          `json:"tool_call_id"`
		ToolCalls  []ToolCall      `json:"tool_calls"`
	}
	_ = json.Unmarshal(raw, &m)
	return decodedMessage{Role: m.Role, Content: contentText(m.Content), ToolCallID: m.ToolCallID, ToolCalls: m.ToolCalls}
}

// contentText extracts text from a string or an array of content parts.
func contentText(c json.RawMessage) string {
	if len(c) == 0 || string(c) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(c, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(c, &parts) == nil {
		var texts []string
		for _, p := range parts {
			if p.Text != "" {
				texts = append(texts, p.Text)
			}
		}
		return strings.Join(texts, "\n")
	}
	return string(c)
}

// newMessage fills the decoded fields of a message from its raw JSON.
func newMessage(sessionID string, raw json.RawMessage, status MessageStatus) Message {
	d := decodeMessage(raw)
	now := time.Now()
	return Message{
		ID: newID("msg"), SessionID: sessionID, Role: d.Role, Status: status, Content: d.Content,
		ToolCalls: d.ToolCalls, ToolCallID: d.ToolCallID, Raw: raw, CreatedAt: now, UpdatedAt: now,
	}
}

// insertMessage assigns the next seq and stores m.
func insertMessage(ctx context.Context, store Store, m *Message) error {
	rows, err := store.Query(ctx, "SELECT COALESCE(MAX(seq), 0) AS max_seq FROM agent_messages WHERE session_id = ?", m.SessionID)
	if err != nil {
		return err
	}
	var seq int64
	if rows.Next() {
		err = rows.Scan(&seq)
	}
	rows.Close()
	if err != nil {
		return err
	}
	m.Seq = seq + 1
	_, err = store.Exec(ctx,
		"INSERT INTO agent_messages ("+messageCols+") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		m.ID, m.SessionID, m.Seq, m.Role, m.Kind, m.RefSeq, string(m.Status), m.Content, m.Reasoning, m.ToolCallID, string(m.Raw),
		m.CreatedAt.UnixMilli(), m.UpdatedAt.UnixMilli())
	if err != nil {
		return fmt.Errorf("agent: insert message: %w", err)
	}
	return nil
}

// updateMessage rewrites the mutable parts of a message (while streaming or
// while its tool call is in progress).
func updateMessage(ctx context.Context, store Store, l lease, m *Message) error {
	d := decodeMessage(m.Raw)
	m.Content, m.ToolCalls = d.Content, d.ToolCalls
	m.UpdatedAt = time.Now()
	q, args := l.fence("UPDATE agent_messages SET status = ?, content = ?, reasoning = ?, raw = ?, updated_at = ? WHERE id = ?",
		[]any{string(m.Status), m.Content, m.Reasoning, string(m.Raw), m.UpdatedAt.UnixMilli(), m.ID})
	_, err := store.Exec(ctx, q, args...)
	if err != nil {
		return fmt.Errorf("agent: update message: %w", err)
	}
	return nil
}

// getMessage loads one message, e.g. to poll a message that is still streaming.
func getMessage(ctx context.Context, store Store, sessionID, messageID string) (*Message, error) {
	rows, err := store.Query(ctx, "SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND id = ?", sessionID, messageID)
	if err != nil {
		return nil, err
	}
	ms, err := scanMessages(rows)
	if err != nil {
		return nil, err
	}
	if len(ms) == 0 {
		return nil, ErrMessageNotFound
	}
	return &ms[0], nil
}

// messagesFromSeq returns the messages with seq >= from, oldest first.
func messagesFromSeq(ctx context.Context, store Store, sessionID string, from int64) ([]Message, error) {
	rows, err := store.Query(ctx, "SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND seq >= ? ORDER BY seq ASC", sessionID, from)
	if err != nil {
		return nil, err
	}
	return scanMessages(rows)
}

// latestCompaction returns the newest compaction message, if any.
func latestCompaction(ctx context.Context, store Store, sessionID string) (*Message, error) {
	rows, err := store.Query(ctx, "SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND kind = ? ORDER BY seq DESC LIMIT 1", sessionID, MessageKindCompaction)
	if err != nil {
		return nil, err
	}
	ms, err := scanMessages(rows)
	if err != nil || len(ms) == 0 {
		return nil, err
	}
	return &ms[0], nil
}

// lastAssistant returns the newest assistant message, if any.
func lastAssistant(ctx context.Context, store Store, sessionID string) (*Message, error) {
	rows, err := store.Query(ctx, "SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND role = 'assistant' ORDER BY seq DESC LIMIT 1", sessionID)
	if err != nil {
		return nil, err
	}
	ms, err := scanMessages(rows)
	if err != nil || len(ms) == 0 {
		return nil, err
	}
	return &ms[0], nil
}

func messagesAfterSeq(ctx context.Context, store Store, sessionID string, seq int64) ([]Message, error) {
	rows, err := store.Query(ctx, "SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND seq > ? ORDER BY seq ASC", sessionID, seq)
	if err != nil {
		return nil, err
	}
	return scanMessages(rows)
}

func normLimit(limit int) int {
	if limit <= 0 {
		return 50
	}
	return limit
}

func reverse(ms []Message) []Message {
	for i, j := 0, len(ms)-1; i < j; i, j = i+1, j-1 {
		ms[i], ms[j] = ms[j], ms[i]
	}
	return ms
}

// latestMessages returns the newest limit messages in chronological order.
func latestMessages(ctx context.Context, store Store, sessionID string, limit int) ([]Message, error) {
	rows, err := store.Query(ctx, fmt.Sprintf("SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? ORDER BY seq DESC LIMIT %d", normLimit(limit)), sessionID)
	if err != nil {
		return nil, err
	}
	ms, err := scanMessages(rows)
	return reverse(ms), err
}

// messagesBefore returns up to limit messages older than messageID, in
// chronological order (for scrolling back through history).
func messagesBefore(ctx context.Context, store Store, sessionID, messageID string, limit int) ([]Message, error) {
	seq, err := messageSeq(ctx, store, sessionID, messageID)
	if err != nil {
		return nil, err
	}
	rows, err := store.Query(ctx, fmt.Sprintf("SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND seq < ? ORDER BY seq DESC LIMIT %d", normLimit(limit)), sessionID, seq)
	if err != nil {
		return nil, err
	}
	ms, err := scanMessages(rows)
	return reverse(ms), err
}

// messagesAfter returns up to limit messages newer than messageID, in
// chronological order.
func messagesAfter(ctx context.Context, store Store, sessionID, messageID string, limit int) ([]Message, error) {
	seq, err := messageSeq(ctx, store, sessionID, messageID)
	if err != nil {
		return nil, err
	}
	rows, err := store.Query(ctx, fmt.Sprintf("SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? AND seq > ? ORDER BY seq ASC LIMIT %d", normLimit(limit)), sessionID, seq)
	if err != nil {
		return nil, err
	}
	return scanMessages(rows)
}

func messageSeq(ctx context.Context, store Store, sessionID, messageID string) (int64, error) {
	rows, err := store.Query(ctx, "SELECT seq FROM agent_messages WHERE session_id = ? AND id = ?", sessionID, messageID)
	if err != nil {
		return 0, err
	}
	defer rows.Close()
	if !rows.Next() {
		if err := rows.Err(); err != nil {
			return 0, err
		}
		return 0, ErrMessageNotFound
	}
	var seq int64
	err = rows.Scan(&seq)
	return seq, err
}
