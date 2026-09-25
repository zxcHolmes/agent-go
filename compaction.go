package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"
)

const defaultCompactionPrompt = "You are compacting a conversation between a user and an AI assistant so that it fits in a limited context window. " +
	"Write concise notes the assistant can rely on to continue the conversation. Preserve the user's goals and preferences, " +
	"key facts, names, ids and numbers, decisions taken, results of tool calls, and any unfinished tasks. " +
	"Write only the notes, no preamble."

// prepareContext returns the messages to send and the output token budget,
// compacting the history first when it has grown past the threshold.
func (a *Agent) prepareContext(ctx context.Context, st *runState) ([]Message, int, error) {
	window, marker, err := a.contextWindow(st)
	if err != nil {
		return nil, 0, err
	}
	est, err := a.estimateTokens(st, window, marker)
	if err != nil {
		return nil, 0, err
	}
	if a.needsCompaction(est) {
		compacted, err := a.compact(ctx, st, window, marker)
		if err != nil {
			return nil, 0, err
		}
		if compacted {
			if window, marker, err = a.contextWindow(st); err != nil {
				return nil, 0, err
			}
			if est, err = a.estimateTokens(st, window, marker); err != nil {
				return nil, 0, err
			}
		}
	}
	budget, err := a.outputBudget(est)
	return window, budget, err
}

// contextWindow returns the messages the model sees: everything from the
// latest compaction's RefSeq on, preceded by its summary in summary mode.
// It also returns that compaction message (nil if the session has none).
func (a *Agent) contextWindow(st *runState) ([]Message, *Message, error) {
	all, err := allMessages(st.db, a.store, a.sessionID)
	if err != nil {
		return nil, nil, err
	}
	var marker *Message
	for i := len(all) - 1; i >= 0; i-- {
		if all[i].Kind == MessageKindCompaction {
			marker = &all[i]
			break
		}
	}
	window := make([]Message, 0, len(all))
	if marker != nil && marker.Status == MessageDone { // a summary to send
		window = append(window, *marker)
	}
	for _, m := range all {
		if m.Kind == MessageKindCompaction || (marker != nil && m.Seq < marker.RefSeq) {
			continue
		}
		window = append(window, m)
	}
	return replayable(window), marker, nil
}

// estimateTokens estimates the prompt size of the next request: the exact
// token count of the previous call (when it saw the same window) plus ~3
// bytes per token for newer messages.
func (a *Agent) estimateTokens(st *runState, window []Message, marker *Message) (int, error) {
	used, afterSeq, ok, err := lastCallSize(st.db, a.store, a.sessionID)
	if err != nil {
		return 0, err
	}
	if ok && marker != nil && afterSeq < marker.Seq {
		ok = false // that call predates the compaction; its size no longer applies
	}
	est := int(used)
	if !ok {
		est = len(a.system) / 3
		for _, t := range a.tools {
			est += len(t) / 3
		}
		afterSeq = -1
	}
	for _, m := range window {
		if m.Seq > afterSeq {
			est += len(m.Raw)/3 + 4
		}
	}
	return est, nil
}

func (a *Agent) needsCompaction(est int) bool {
	return a.cfg.ContextLength > 0 && a.cfg.Compaction != CompactionOff &&
		float64(est) >= a.cfg.CompactionThreshold*float64(a.cfg.ContextLength)
}

// outputBudget returns max_tokens for the request, failing when the prompt
// alone no longer fits.
func (a *Agent) outputBudget(est int) (int, error) {
	out := a.cfg.MaxOutputTokens
	if a.cfg.ContextLength <= 0 {
		return out, nil
	}
	remaining := a.cfg.ContextLength - est
	if remaining <= 0 {
		return 0, fmt.Errorf("%w (estimated %d of %d tokens)", ErrContextLengthExceeded, est, a.cfg.ContextLength)
	}
	if out <= 0 || out > remaining {
		out = remaining
	}
	return out, nil
}

// compact moves the start of the context to the most recent user message
// and records a compaction message; in summary mode the dropped messages are
// summarized first. It reports false when there is nothing to drop (the
// current turn already starts the window).
func (a *Agent) compact(ctx context.Context, st *runState, window []Message, marker *Message) (bool, error) {
	var anchor *Message
	for i := len(window) - 1; i >= 0; i-- {
		if m := window[i]; m.Role == "user" && m.Kind == "" {
			anchor = &window[i]
			break
		}
	}
	firstSeq := int64(-1)
	for _, m := range window {
		if m.Kind != MessageKindCompaction {
			firstSeq = m.Seq
			break
		}
	}
	if anchor == nil || anchor.Seq <= firstSeq {
		return false, nil
	}
	var dropped []Message
	for _, m := range window {
		if m.Seq < anchor.Seq || m.Kind == MessageKindCompaction {
			dropped = append(dropped, m)
		}
	}

	m := Message{
		ID: newID("msg"), SessionID: a.sessionID, Kind: MessageKindCompaction, RefSeq: anchor.Seq,
		Status: MessageExcluded, CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}
	note := fmt.Sprintf("Context compacted: messages before #%d are no longer sent to the model.", anchor.Seq)
	var rec *UsageRecord
	if a.cfg.Compaction == CompactionSummary {
		summary, r, err := a.summarize(ctx, st, dropped, m.ID)
		if err != nil {
			if ctx.Err() != nil {
				return false, err // stopped: leave the history as it was
			}
			// Keep the conversation going without a summary rather than failing it.
			note += " (summary failed: " + truncate(err.Error(), 200) + ")"
		} else {
			rec = r
			raw, _ := userRaw("<conversation_summary>\nThe earlier part of this conversation was compacted. Notes:\n" + summary + "\n</conversation_summary>")
			m.Role, m.Raw, m.Content, m.Status = "user", raw, summary, MessageDone
		}
	}
	if m.Raw == nil {
		m.Raw, _ = marshalJSON(struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"system", note})
		m.Role, m.Content = "system", note
	}
	if err := a.insertMessage(ctx, st, &m); err != nil {
		return false, err
	}
	if rec != nil {
		if err := a.recordUsage(ctx, st, rec); err != nil {
			return false, err
		}
	}
	return true, nil
}

// summarize asks the model for notes on the dropped messages.
func (a *Agent) summarize(ctx context.Context, st *runState, dropped []Message, messageID string) (string, *UsageRecord, error) {
	prompt := a.cfg.CompactionPrompt
	if prompt == "" {
		prompt = defaultCompactionPrompt
	}
	maxTokens := a.cfg.ContextLength / 4
	if maxTokens > 2048 {
		maxTokens = 2048
	}
	// Same ~3 bytes/token estimate as the rest of the SDK, leaving room for
	// the instructions and the summary itself.
	budget := (a.cfg.ContextLength - maxTokens - len(prompt)/3 - 200) * 3
	system, _ := marshalJSON(map[string]string{"role": "system", "content": prompt})
	user, _ := userRaw("Conversation to summarize:\n\n" + transcript(dropped, budget))
	req := chatRequest{Model: a.cfg.Model, Messages: []json.RawMessage{system, user}}
	if a.cfg.UseMaxCompletionTokens {
		req.MaxCompletionTokens = maxTokens
	} else {
		req.MaxTokens = maxTokens
	}
	start := time.Now()
	res, err := a.llm.stream(ctx, req, a.cfg.ExtraBody, nil)
	if err != nil {
		return "", nil, err
	}
	summary := strings.TrimSpace(res.Content)
	if summary == "" {
		return "", nil, fmt.Errorf("agent: empty summary")
	}
	usage := parseUsage(res.Usage)
	rec := &UsageRecord{
		ID: newID("llm"), SessionID: a.sessionID, MessageID: messageID, Model: a.cfg.Model,
		ResponseID: res.ID, FinishReason: res.FinishReason, Usage: usage, RawUsage: res.Usage,
		LatencyMs: time.Since(start).Milliseconds(), CreatedAt: time.Now(),
	}
	if a.cfg.Billing != nil {
		rec.Cost = a.cfg.Billing.Cost(a.cfg.Model, usage)
	}
	return summary, rec, nil
}

// transcript renders messages as plain text for summarization within
// maxBytes. Long messages are trimmed individually first so every turn stays
// represented; if it is still too long, the middle is cut, keeping the start
// (earlier notes, first facts) and the most recent part.
func transcript(msgs []Message, maxBytes int) string {
	var parts []string
	for _, m := range msgs {
		clip := func(s string, n int) string {
			if cut, dropped := truncateRunes(s, n); dropped > 0 {
				return cut + fmt.Sprintf("…[%d characters omitted]", dropped)
			}
			return s
		}
		var p string
		switch {
		case m.Kind == MessageKindCompaction:
			p = "Earlier notes:\n" + m.Content
		case m.Kind == MessageKindViewImage:
			p = "[image shown to the assistant: " + m.Content + "]"
		case m.Role == "user":
			p = "User: " + clip(m.Content, 2000)
		case m.Role == "assistant":
			p = "Assistant: " + clip(m.Content, 1200)
			for _, tc := range m.ToolCalls {
				p += "\n[called " + tc.Function.Name + " " + clip(tc.Function.Arguments, 300) + "]"
			}
		case m.Role == "tool":
			p = "[tool result: " + clip(m.Content, 800) + "]"
		default:
			continue
		}
		parts = append(parts, p)
	}
	s := strings.Join(parts, "\n\n")
	if maxBytes <= 0 || len(s) <= maxBytes {
		return s
	}
	const gap = "\n\n…(middle of the conversation omitted)…\n\n"
	head, tail := maxBytes*3/10, maxBytes*7/10-len(gap)
	return strings.ToValidUTF8(s[:head], "") + gap + strings.ToValidUTF8(s[len(s)-tail:], "")
}
