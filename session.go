package agent

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

func newID(prefix string) string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return prefix + "_" + hex.EncodeToString(b[:])
}

func nowMillis() int64 { return time.Now().UnixMilli() }

func fromMillis(ms int64) time.Time { return time.UnixMilli(ms) }

func rawOrNil(s string) json.RawMessage {
	if s == "" {
		return nil
	}
	return json.RawMessage(s)
}

// ---- sessions ----

// CreateSession creates a new idle session and returns its id. metadata is
// optional and stored as JSON (e.g. the owning user id).
func CreateSession(ctx context.Context, store Store, metadata map[string]any) (string, error) {
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
		"INSERT INTO agent_sessions (id, status, run_id, stop_requested, last_error, metadata, created_at, updated_at) VALUES (?, ?, '', 0, '', ?, ?, ?)",
		id, string(StatusIdle), string(meta), now, now)
	if err != nil {
		return "", fmt.Errorf("agent: create session: %w", err)
	}
	return id, nil
}

// GetSession loads a session.
func GetSession(ctx context.Context, store Store, id string) (*Session, error) {
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

// ResetSession forcibly releases a session left "running" by a crashed
// process. Only use it when you know no run is active. Calls still awaiting
// confirmation keep the session in StatusWaitingConfirmation.
func ResetSession(ctx context.Context, store Store, id string) error {
	pending, err := PendingCalls(ctx, store, id)
	if err != nil {
		return err
	}
	status := StatusIdle
	if len(pending) > 0 {
		status = StatusWaitingConfirmation
	}
	_, err = store.Exec(ctx, "UPDATE agent_sessions SET status = ?, run_id = '', stop_requested = 0, updated_at = ? WHERE id = ?",
		string(status), nowMillis(), id)
	return err
}

// ---- messages ----

const messageCols = "id, session_id, seq, role, content, tool_call_id, raw, created_at"

func scanMessages(rows Rows) ([]Message, error) {
	defer rows.Close()
	var out []Message
	for rows.Next() {
		var m Message
		var raw string
		var created int64
		if err := rows.Scan(&m.ID, &m.SessionID, &m.Seq, &m.Role, &m.Content, &m.ToolCallID, &raw, &created); err != nil {
			return nil, err
		}
		m.Raw = json.RawMessage(raw)
		m.CreatedAt = fromMillis(created)
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

func insertMessage(ctx context.Context, store Store, sessionID string, raw json.RawMessage) (Message, error) {
	d := decodeMessage(raw)
	rows, err := store.Query(ctx, "SELECT COALESCE(MAX(seq), 0) FROM agent_messages WHERE session_id = ?", sessionID)
	if err != nil {
		return Message{}, err
	}
	var seq int64
	if rows.Next() {
		err = rows.Scan(&seq)
	}
	rows.Close()
	if err != nil {
		return Message{}, err
	}
	m := Message{
		ID:         newID("msg"),
		SessionID:  sessionID,
		Seq:        seq + 1,
		Role:       d.Role,
		Content:    d.Content,
		ToolCalls:  d.ToolCalls,
		ToolCallID: d.ToolCallID,
		Raw:        raw,
		CreatedAt:  time.Now(),
	}
	_, err = store.Exec(ctx,
		"INSERT INTO agent_messages ("+messageCols+") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
		m.ID, m.SessionID, m.Seq, m.Role, m.Content, m.ToolCallID, string(m.Raw), m.CreatedAt.UnixMilli())
	if err != nil {
		return Message{}, fmt.Errorf("agent: insert message: %w", err)
	}
	return m, nil
}

func allMessages(ctx context.Context, store Store, sessionID string) ([]Message, error) {
	rows, err := store.Query(ctx, "SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? ORDER BY seq ASC", sessionID)
	if err != nil {
		return nil, err
	}
	return scanMessages(rows)
}

func lastMessage(ctx context.Context, store Store, sessionID string) (*Message, error) {
	ms, err := LatestMessages(ctx, store, sessionID, 1)
	if err != nil || len(ms) == 0 {
		return nil, err
	}
	return &ms[0], nil
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

// LatestMessages returns the newest limit messages in chronological order.
func LatestMessages(ctx context.Context, store Store, sessionID string, limit int) ([]Message, error) {
	rows, err := store.Query(ctx, fmt.Sprintf("SELECT "+messageCols+" FROM agent_messages WHERE session_id = ? ORDER BY seq DESC LIMIT %d", normLimit(limit)), sessionID)
	if err != nil {
		return nil, err
	}
	ms, err := scanMessages(rows)
	return reverse(ms), err
}

// MessagesBefore returns up to limit messages older than messageID, in
// chronological order (for scrolling back through history).
func MessagesBefore(ctx context.Context, store Store, sessionID, messageID string, limit int) ([]Message, error) {
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

// MessagesAfter returns up to limit messages newer than messageID, in
// chronological order.
func MessagesAfter(ctx context.Context, store Store, sessionID, messageID string, limit int) ([]Message, error) {
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

// ---- rpc calls ----

const callCols = "id, session_id, message_id, tool_call_id, call_index, method, params, rpc_id, require_confirm, status, result, result_message_id, created_at, updated_at"

func scanCalls(rows Rows) ([]RPCCall, error) {
	defer rows.Close()
	var out []RPCCall
	for rows.Next() {
		var c RPCCall
		var idx, confirm, created, updated int64
		var params, rpcID, status, result string
		if err := rows.Scan(&c.ID, &c.SessionID, &c.MessageID, &c.ToolCallID, &idx, &c.Method, &params, &rpcID, &confirm, &status, &result, &c.ResultMessageID, &created, &updated); err != nil {
			return nil, err
		}
		c.Index = int(idx)
		c.Params, c.RPCID, c.Result = rawOrNil(params), rawOrNil(rpcID), rawOrNil(result)
		c.RequireConfirm = confirm != 0
		c.Status = CallStatus(status)
		c.CreatedAt, c.UpdatedAt = fromMillis(created), fromMillis(updated)
		out = append(out, c)
	}
	return out, rows.Err()
}

func boolInt(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

func insertCall(ctx context.Context, store Store, c *RPCCall) error {
	_, err := store.Exec(ctx,
		"INSERT INTO agent_rpc_calls ("+callCols+") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		c.ID, c.SessionID, c.MessageID, c.ToolCallID, int64(c.Index), c.Method, string(c.Params), string(c.RPCID),
		boolInt(c.RequireConfirm), string(c.Status), string(c.Result), c.ResultMessageID, c.CreatedAt.UnixMilli(), c.UpdatedAt.UnixMilli())
	if err != nil {
		return fmt.Errorf("agent: insert rpc call: %w", err)
	}
	return nil
}

func updateCall(ctx context.Context, store Store, c *RPCCall) error {
	c.UpdatedAt = time.Now()
	_, err := store.Exec(ctx, "UPDATE agent_rpc_calls SET status = ?, result = ?, result_message_id = ?, updated_at = ? WHERE id = ?",
		string(c.Status), string(c.Result), c.ResultMessageID, c.UpdatedAt.UnixMilli(), c.ID)
	if err != nil {
		return fmt.Errorf("agent: update rpc call: %w", err)
	}
	return nil
}

// openCalls returns calls whose tool result message has not been written yet.
func openCalls(ctx context.Context, store Store, sessionID string) ([]RPCCall, error) {
	rows, err := store.Query(ctx, "SELECT "+callCols+" FROM agent_rpc_calls WHERE session_id = ? AND result_message_id = '' ORDER BY created_at ASC, call_index ASC", sessionID)
	if err != nil {
		return nil, err
	}
	return scanCalls(rows)
}

func callsForMessage(ctx context.Context, store Store, messageID string) ([]RPCCall, error) {
	rows, err := store.Query(ctx, "SELECT "+callCols+" FROM agent_rpc_calls WHERE message_id = ? ORDER BY call_index ASC", messageID)
	if err != nil {
		return nil, err
	}
	return scanCalls(rows)
}

// PendingCalls returns the RPC calls of a session waiting for confirmation.
func PendingCalls(ctx context.Context, store Store, sessionID string) ([]RPCCall, error) {
	rows, err := store.Query(ctx, "SELECT "+callCols+" FROM agent_rpc_calls WHERE session_id = ? AND status = ? ORDER BY created_at ASC, call_index ASC",
		sessionID, string(CallAwaitingConfirmation))
	if err != nil {
		return nil, err
	}
	return scanCalls(rows)
}

// ---- usage ----

const usageCols = "id, session_id, message_id, model, response_id, finish_reason, prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_write_tokens, reasoning_tokens, audio_input_tokens, audio_output_tokens, accepted_prediction_tokens, rejected_prediction_tokens, raw_usage, input_credits, cached_credits, cache_write_credits, output_credits, total_credits, latency_ms, created_at"

func insertUsage(ctx context.Context, store Store, r *UsageRecord) error {
	u, c := r.Usage, r.Cost
	_, err := store.Exec(ctx,
		"INSERT INTO agent_llm_calls ("+usageCols+") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
		r.ID, r.SessionID, r.MessageID, r.Model, r.ResponseID, r.FinishReason,
		u.PromptTokens, u.CompletionTokens, u.TotalTokens, u.CachedTokens, u.CacheWriteTokens, u.ReasoningTokens,
		u.AudioInputTokens, u.AudioOutputTokens, u.AcceptedPredictionTokens, u.RejectedPredictionTokens,
		string(r.RawUsage), c.Input, c.CachedInput, c.CacheWrite, c.Output, c.Total, r.LatencyMs, r.CreatedAt.UnixMilli())
	if err != nil {
		return fmt.Errorf("agent: insert usage: %w", err)
	}
	return nil
}

// ListUsage returns every LLM call of a session, oldest first.
func ListUsage(ctx context.Context, store Store, sessionID string) ([]UsageRecord, error) {
	rows, err := store.Query(ctx, "SELECT "+usageCols+" FROM agent_llm_calls WHERE session_id = ? ORDER BY created_at ASC", sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []UsageRecord
	for rows.Next() {
		var r UsageRecord
		var raw string
		var created int64
		u, c := &r.Usage, &r.Cost
		if err := rows.Scan(&r.ID, &r.SessionID, &r.MessageID, &r.Model, &r.ResponseID, &r.FinishReason,
			&u.PromptTokens, &u.CompletionTokens, &u.TotalTokens, &u.CachedTokens, &u.CacheWriteTokens, &u.ReasoningTokens,
			&u.AudioInputTokens, &u.AudioOutputTokens, &u.AcceptedPredictionTokens, &u.RejectedPredictionTokens,
			&raw, &c.Input, &c.CachedInput, &c.CacheWrite, &c.Output, &c.Total, &r.LatencyMs, &created); err != nil {
			return nil, err
		}
		r.RawUsage = rawOrNil(raw)
		r.CreatedAt = fromMillis(created)
		out = append(out, r)
	}
	return out, rows.Err()
}

// SessionUsage sums token usage and credits over all LLM calls of a session.
func SessionUsage(ctx context.Context, store Store, sessionID string) (*UsageSummary, error) {
	rows, err := store.Query(ctx, `SELECT COUNT(*),
		COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(completion_tokens), 0), COALESCE(SUM(total_tokens), 0),
		COALESCE(SUM(cached_tokens), 0), COALESCE(SUM(cache_write_tokens), 0), COALESCE(SUM(reasoning_tokens), 0),
		COALESCE(SUM(audio_input_tokens), 0), COALESCE(SUM(audio_output_tokens), 0),
		COALESCE(SUM(accepted_prediction_tokens), 0), COALESCE(SUM(rejected_prediction_tokens), 0),
		COALESCE(SUM(input_credits), 0), COALESCE(SUM(cached_credits), 0), COALESCE(SUM(cache_write_credits), 0),
		COALESCE(SUM(output_credits), 0), COALESCE(SUM(total_credits), 0)
		FROM agent_llm_calls WHERE session_id = ?`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var s UsageSummary
	if rows.Next() {
		u, c := &s.Usage, &s.Cost
		if err := rows.Scan(&s.Calls, &u.PromptTokens, &u.CompletionTokens, &u.TotalTokens, &u.CachedTokens, &u.CacheWriteTokens,
			&u.ReasoningTokens, &u.AudioInputTokens, &u.AudioOutputTokens, &u.AcceptedPredictionTokens, &u.RejectedPredictionTokens,
			&c.Input, &c.CachedInput, &c.CacheWrite, &c.Output, &c.Total); err != nil {
			return nil, err
		}
	}
	return &s, rows.Err()
}

// lastCallSize returns prompt+completion tokens of the latest LLM call and the
// seq of the assistant message it produced.
func lastCallSize(ctx context.Context, store Store, sessionID string) (tokens, seq int64, ok bool, err error) {
	rows, err := store.Query(ctx, `SELECT c.prompt_tokens + c.completion_tokens, m.seq
		FROM agent_llm_calls c JOIN agent_messages m ON m.id = c.message_id
		WHERE c.session_id = ? ORDER BY m.seq DESC LIMIT 1`, sessionID)
	if err != nil {
		return 0, 0, false, err
	}
	defer rows.Close()
	if !rows.Next() {
		return 0, 0, false, rows.Err()
	}
	err = rows.Scan(&tokens, &seq)
	return tokens, seq, err == nil, err
}
