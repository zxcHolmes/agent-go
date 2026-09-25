package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"time"
)

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

// openCalls returns calls that have not finished yet.
func openCalls(ctx context.Context, store Store, sessionID string) ([]RPCCall, error) {
	rows, err := store.Query(ctx, "SELECT "+callCols+" FROM agent_rpc_calls WHERE session_id = ? AND status IN (?, ?, ?, ?) ORDER BY created_at ASC, call_index ASC",
		sessionID, string(CallQueued), string(CallAwaitingConfirmation), string(CallApproved), string(CallRunning))
	if err != nil {
		return nil, err
	}
	return scanCalls(rows)
}

func toolMessageRaw(toolCallID, content string) json.RawMessage {
	raw, _ := marshalJSON(struct {
		Role       string `json:"role"`
		ToolCallID string `json:"tool_call_id"`
		Content    string `json:"content"`
	}{"tool", toolCallID, content})
	return raw
}

// setCallStatus moves a call to a non-final status and mirrors it on its tool message.
func setCallStatus(ctx context.Context, store Store, c *RPCCall, status CallStatus) error {
	c.Status = status
	if err := updateCall(ctx, store, c); err != nil {
		return err
	}
	ms := MessagePending
	if status == CallRunning {
		ms = MessageRunning
	}
	_, err := store.Exec(ctx, "UPDATE agent_messages SET status = ?, updated_at = ? WHERE id = ?", string(ms), nowMillis(), c.ResultMessageID)
	return err
}

// completeCall stores the final result of a call and writes it into its tool message.
func completeCall(ctx context.Context, store Store, c *RPCCall, status CallStatus, result json.RawMessage) error {
	c.Status, c.Result = status, result
	if err := updateCall(ctx, store, c); err != nil {
		return err
	}
	if c.ResultMessageID == "" {
		return nil
	}
	m := Message{ID: c.ResultMessageID, Status: MessageDone, Raw: toolMessageRaw(c.ToolCallID, toolContent(result))}
	return updateMessage(ctx, store, &m)
}

func callsForMessage(ctx context.Context, store Store, messageID string) ([]RPCCall, error) {
	rows, err := store.Query(ctx, "SELECT "+callCols+" FROM agent_rpc_calls WHERE message_id = ? ORDER BY call_index ASC", messageID)
	if err != nil {
		return nil, err
	}
	return scanCalls(rows)
}

// pendingCalls returns the RPC calls of a session waiting for confirmation.
func pendingCalls(ctx context.Context, store Store, sessionID string) ([]RPCCall, error) {
	rows, err := store.Query(ctx, "SELECT "+callCols+" FROM agent_rpc_calls WHERE session_id = ? AND status = ? ORDER BY created_at ASC, call_index ASC",
		sessionID, string(CallAwaitingConfirmation))
	if err != nil {
		return nil, err
	}
	return scanCalls(rows)
}
