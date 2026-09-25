package agent

import (
	"context"
	"fmt"
)

const usageCols = "id, session_id, message_id, model, response_id, finish_reason, prompt_tokens, completion_tokens, total_tokens, cached_tokens, cache_write_tokens, reasoning_tokens, audio_input_tokens, audio_output_tokens, accepted_prediction_tokens, rejected_prediction_tokens, raw_usage, input_credits, cached_credits, cache_write_credits, output_credits, total_credits, latency_ms, bill_id, claimed_at, billed_at, created_at"

func insertUsage(ctx context.Context, store Store, r *UsageRecord) error {
	u, c := r.Usage, r.Cost
	_, err := store.Exec(ctx,
		"INSERT INTO agent_llm_calls ("+usageCols+") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', 0, 0, ?)",
		r.ID, r.SessionID, r.MessageID, r.Model, r.ResponseID, r.FinishReason,
		u.PromptTokens, u.CompletionTokens, u.TotalTokens, u.CachedTokens, u.CacheWriteTokens, u.ReasoningTokens,
		u.AudioInputTokens, u.AudioOutputTokens, u.AcceptedPredictionTokens, u.RejectedPredictionTokens,
		string(r.RawUsage), c.Input, c.CachedInput, c.CacheWrite, c.Output, c.Total, r.LatencyMs, r.CreatedAt.UnixMilli())
	if err != nil {
		return fmt.Errorf("agent: insert usage: %w", err)
	}
	return nil
}

func queryUsage(ctx context.Context, store Store, where string, args ...any) ([]UsageRecord, error) {
	rows, err := store.Query(ctx, "SELECT "+usageCols+" FROM agent_llm_calls WHERE "+where+" ORDER BY created_at ASC", args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []UsageRecord
	for rows.Next() {
		var r UsageRecord
		var raw string
		var claimed, billed, created int64
		u, c := &r.Usage, &r.Cost
		if err := rows.Scan(&r.ID, &r.SessionID, &r.MessageID, &r.Model, &r.ResponseID, &r.FinishReason,
			&u.PromptTokens, &u.CompletionTokens, &u.TotalTokens, &u.CachedTokens, &u.CacheWriteTokens, &u.ReasoningTokens,
			&u.AudioInputTokens, &u.AudioOutputTokens, &u.AcceptedPredictionTokens, &u.RejectedPredictionTokens,
			&raw, &c.Input, &c.CachedInput, &c.CacheWrite, &c.Output, &c.Total, &r.LatencyMs,
			&r.BillID, &claimed, &billed, &created); err != nil {
			return nil, err
		}
		r.RawUsage = rawOrNil(raw)
		r.CreatedAt = fromMillis(created)
		if claimed > 0 {
			t := fromMillis(claimed)
			r.ClaimedAt = &t
		}
		if billed > 0 {
			t := fromMillis(billed)
			r.BilledAt = &t
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// listUsage returns every LLM call of a session, oldest first.
func listUsage(ctx context.Context, store Store, sessionID string) ([]UsageRecord, error) {
	return queryUsage(ctx, store, "session_id = ?", sessionID)
}

// sessionUsage sums token usage and credits over all LLM calls of a session.
func sessionUsage(ctx context.Context, store Store, sessionID string) (*UsageSummary, error) {
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
		WHERE c.session_id = ? AND m.role = 'assistant' ORDER BY m.seq DESC LIMIT 1`, sessionID)
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
