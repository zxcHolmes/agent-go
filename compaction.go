package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"
)

// defaultCompactionRequest is the user message appended to the context when
// the model writes its own notes (see summarize).
const defaultCompactionRequest = "Context checkpoint: everything above is about to be removed from your context. " +
	"Only your notes and the latest user request, with what follows it, will remain. " +
	"Write the notes you need to continue: the user's goals and preferences, key facts, names, and every id, URL and number still needed, copied exactly; " +
	"decisions taken, results of tool calls, and any unfinished tasks. " +
	"Do not call any tool and do not address the user. Write only the notes."

// prepareContext returns the messages to send and the output token budget,
// compacting the history first when it has grown past the threshold.
func (a *Agent) prepareContext(ctx context.Context, st *runState) (window []Message, budget, est int, err error) {
	window, marker, err := a.contextWindow(st)
	if err != nil {
		return nil, 0, 0, err
	}
	if est, err = a.estimateTokens(st, window, marker); err != nil {
		return nil, 0, 0, err
	}
	if a.needsCompaction(est) {
		before := est
		compacted, err := a.compact(ctx, st, window, marker, est)
		if err != nil {
			return nil, 0, 0, err
		}
		if compacted {
			if window, marker, err = a.contextWindow(st); err != nil {
				return nil, 0, 0, err
			}
			if est, err = a.estimateTokens(st, window, marker); err != nil {
				return nil, 0, 0, err
			}
			a.log.Info("context compacted", "session", a.sessionID, "mode", string(a.cfg.Compaction),
				"restart_at_seq", marker.RefSeq, "estimated_tokens_before", before, "estimated_tokens_after", est)
		} else {
			a.log.Debug("compaction skipped: the current turn already starts the context", "session", a.sessionID, "estimated_tokens", est)
		}
	}
	budget, err = a.outputBudget(est)
	return window, budget, est, err
}

// rejectedError marks a BeforeLLMCall rejection so compaction does not
// swallow it like an ordinary summary failure.
type rejectedError struct{ err error }

func (e *rejectedError) Error() string { return e.err.Error() }
func (e *rejectedError) Unwrap() error { return e.err }

// contextWindow returns the messages the model sees: everything from the
// latest compaction's RefSeq on, preceded by its summary in summary mode.
// It also returns that compaction message (nil if the session has none).
//
// Only the messages from that point on are read from the store (two indexed
// queries), so the cost of a step does not grow with the session's history.
func (a *Agent) contextWindow(st *runState) ([]Message, *Message, error) {
	marker, err := latestCompaction(st.db, a.store, a.sessionID)
	if err != nil {
		return nil, nil, err
	}
	var from int64
	if marker != nil {
		from = marker.RefSeq
	}
	msgs, err := messagesFromSeq(st.db, a.store, a.sessionID, from)
	if err != nil {
		return nil, nil, err
	}
	window := make([]Message, 0, len(msgs)+1)
	if marker != nil && marker.Status == MessageDone { // a summary to send
		window = append(window, *marker)
	}
	for _, m := range msgs {
		if m.Kind != MessageKindCompaction {
			window = append(window, m)
		}
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
			est += messageTokens(m.Raw)
		}
	}
	return est, nil
}

// imageTokens is the flat estimate for one image_url part.
const imageTokens = 500

// messageTokens estimates one message: ~3 bytes per token for its JSON,
// except image URLs, which count as imageTokens each whatever their length.
func messageTokens(raw json.RawMessage) int {
	n := len(raw)
	var m struct {
		Content []userContentPart `json:"content"`
	}
	if json.Unmarshal(raw, &m) == nil {
		for _, p := range m.Content {
			if p.Type == "image_url" && p.ImageURL != nil {
				n -= len(p.ImageURL.URL)
				n += imageTokens * 3
			}
		}
	}
	return n/3 + 4
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
func (a *Agent) compact(ctx context.Context, st *runState, window []Message, marker *Message, est int) (bool, error) {
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
	m := Message{
		ID: newID("msg"), SessionID: a.sessionID, Kind: MessageKindCompaction, RefSeq: anchor.Seq,
		Status: MessageExcluded, CreatedAt: time.Now(), UpdatedAt: time.Now(),
	}
	note := fmt.Sprintf("Context compacted: messages before #%d are no longer sent to the model.", anchor.Seq)
	var rec *UsageRecord
	if a.cfg.Compaction == CompactionSummary {
		summary, r, err := a.summarize(ctx, st, window, est, m.ID)
		if err != nil {
			var rej *rejectedError
			if errors.As(err, &rej) {
				return false, rej.err
			}
			if ctx.Err() != nil {
				return false, err // stopped: leave the history as it was
			}
			a.log.Warn("summary failed, compacting without it", "session", a.sessionID, "error", err)
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

// summaryMaxTokens caps the notes a summary call may write.
const summaryMaxTokens = 4096

// summarize asks the model for notes on what compaction drops.
//
// It sends the request the next step would send — same system prompt, tools
// and history, byte for byte — with one user message appended asking for
// notes. That prefix is what the provider has cached, so the call pays for the
// instruction and the notes, not the whole context again. The instruction is
// a user message, not a second system message: many chat templates refuse a
// system message anywhere but first. ContextLength is the host's budget, not
// the model's limit, so the request is not checked against it; if the
// provider does refuse it, compaction goes ahead without notes (see compact).
func (a *Agent) summarize(ctx context.Context, st *runState, window []Message, est int, messageID string) (string, *UsageRecord, error) {
	request := a.cfg.CompactionPrompt
	if request == "" {
		request = defaultCompactionRequest
	}
	ask, err := userRaw(request)
	if err != nil {
		return "", nil, err
	}
	msgs := make([]json.RawMessage, 0, len(window)+2)
	if a.system != nil {
		msgs = append(msgs, a.system)
	}
	for _, m := range window {
		msgs = append(msgs, m.Raw)
	}
	msgs = append(msgs, ask)
	if a.cfg.CacheControl {
		msgs = withCacheControl(msgs)
	}
	// The tools stay in the request: they are part of the cached prefix.
	req := chatRequest{Model: a.cfg.Model, Messages: msgs, Tools: a.tools}
	return a.summaryCall(ctx, req, est+len(ask)/3, summaryMaxTokens, messageID)
}

// summaryCall sends one summary request and records its usage against the
// compaction message.
func (a *Agent) summaryCall(ctx context.Context, req chatRequest, est, maxTokens int, messageID string) (string, *UsageRecord, error) {
	if a.cfg.UseMaxCompletionTokens {
		req.MaxCompletionTokens = maxTokens
	} else {
		req.MaxTokens = maxTokens
	}
	if err := a.beforeLLMCall(ctx, "summary", est, maxTokens); err != nil {
		return "", nil, &rejectedError{err}
	}
	start := time.Now()
	res, err := a.llm.stream(ctx, req, a.cfg.ExtraBody, nil)
	if err != nil {
		return "", nil, err
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
	summary := strings.TrimSpace(res.Content)
	if summary == "" {
		return "", nil, fmt.Errorf("agent: empty summary")
	}
	a.log.Debug("summary written", "session", a.sessionID, "prompt_tokens", usage.PromptTokens,
		"cached_tokens", usage.CachedTokens, "completion_tokens", usage.CompletionTokens)
	return summary, rec, nil
}
