package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

// assistantRaw builds the assistant message stored and replayed to the model.
func (a *Agent) assistantRaw(content, reasoning string, calls []ToolCall) json.RawMessage {
	msg := struct {
		Role             string     `json:"role"`
		Content          *string    `json:"content"`
		ReasoningContent string     `json:"reasoning_content,omitempty"`
		ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	}{Role: "assistant", ToolCalls: calls}
	if content != "" || len(calls) == 0 {
		msg.Content = &content
	}
	if a.cfg.KeepReasoning {
		msg.ReasoningContent = reasoning
	}
	for i := range msg.ToolCalls {
		if msg.ToolCalls[i].Type == "" {
			msg.ToolCalls[i].Type = "function"
		}
		if strings.TrimSpace(msg.ToolCalls[i].Function.Arguments) == "" {
			msg.ToolCalls[i].Function.Arguments = "{}"
		}
	}
	raw, _ := marshalJSON(msg)
	return raw
}

// step performs one streamed LLM call. The assistant message is created on
// the first delta, flushed every StreamFlushInterval, and finalised when the
// stream ends (or marked interrupted if it breaks).
func (a *Agent) step(ctx context.Context, st *runState) (bool, error) {
	if _, err := a.drainQueue(ctx, st); err != nil {
		return false, err
	}
	history, maxTokens, est, err := a.prepareContext(ctx, st)
	if err != nil {
		return false, err
	}
	if err := a.beforeLLMCall(ctx, "chat", est, maxTokens); err != nil {
		return false, err
	}
	msgs := make([]json.RawMessage, 0, len(history)+1)
	if a.system != nil {
		msgs = append(msgs, a.system)
	}
	for _, m := range history {
		msgs = append(msgs, m.Raw)
	}
	if a.cfg.CacheControl {
		msgs = withCacheControl(msgs)
	}
	req := chatRequest{Model: a.cfg.Model, Messages: msgs, Tools: a.tools}
	a.log.Debug("llm request", "session", a.sessionID, "model", a.cfg.Model, "messages", len(msgs),
		"estimated_prompt_tokens", est, "max_tokens", maxTokens)
	if a.cfg.UseMaxCompletionTokens {
		req.MaxCompletionTokens = maxTokens
	} else {
		req.MaxTokens = maxTokens
	}

	// Deltas arrive on this goroutine; a trailing timer flushes buffered text
	// when the stream stalls, so the store never lags more than one interval.
	var (
		mu                 sync.Mutex
		msg                *Message
		content, reasoning strings.Builder
		lastFlush          time.Time
		timer              *time.Timer
		dirty, ended       bool
		flushErr           error
	)
	flush := func(status MessageStatus, raw json.RawMessage) error { // mu held
		lastFlush, dirty = time.Now(), false
		if msg == nil {
			m := newMessage(a.sessionID, raw, status)
			m.Reasoning = reasoning.String()
			msg = &m
			return a.insertMessage(ctx, st, msg)
		}
		msg.Raw, msg.Status, msg.Reasoning = raw, status, reasoning.String()
		return a.updateMessage(ctx, st, msg)
	}
	partial := func() json.RawMessage { return a.assistantRaw(content.String(), "", nil) }
	onDelta := func(dc, dr string) error {
		mu.Lock()
		defer mu.Unlock()
		if flushErr != nil {
			return flushErr
		}
		content.WriteString(dc)
		reasoning.WriteString(dr)
		dirty = true
		if wait := a.cfg.StreamFlushInterval - time.Since(lastFlush); msg == nil || wait <= 0 {
			if err := flush(MessageStreaming, partial()); err != nil {
				return err
			}
		} else if timer == nil {
			timer = time.AfterFunc(wait, func() {
				mu.Lock()
				defer mu.Unlock()
				timer = nil
				if dirty && !ended {
					flushErr = flush(MessageStreaming, partial())
				}
			})
		}
		if a.cfg.OnStream != nil {
			a.callback("OnStream", func() { a.cfg.OnStream(ctx, msg.ID, dc, dr) })
		}
		return nil
	}
	endStream := func() { // stop the trailing timer; afterwards only this goroutine touches state
		mu.Lock()
		ended = true
		if timer != nil {
			timer.Stop()
		}
		mu.Unlock()
	}

	start := time.Now()
	var spend streamSpend
	res, err := a.llm.streamSpend(ctx, req, a.cfg.ExtraBody, onDelta, &spend)
	endStream()
	if err != nil && ctx.Err() == nil {
		a.log.Warn("llm request failed", "session", a.sessionID, "error", err)
	}
	mu.Lock() // pairs with a timer callback that may have just run
	defer mu.Unlock()
	if err != nil {
		messageID := ""
		if msg != nil {
			messageID = msg.ID
			// Keep what was generated; partial tool calls are dropped.
			if ferr := flush(MessageInterrupted, partial()); ferr != nil {
				err = errors.Join(err, ferr)
			}
		}
		if rerr := a.recordInterrupted(ctx, st, spend, est, st.prefix, messageID); rerr != nil {
			err = errors.Join(err, rerr)
		}
		return false, err
	}
	latency := time.Since(start).Milliseconds()
	if err := a.ownsLock(st); err != nil {
		return false, err // recovered while streaming; leave the message interrupted
	}
	content.Reset()
	content.WriteString(res.Content)
	reasoning.Reset()
	reasoning.WriteString(res.Reasoning)
	if err := flush(MessageDone, a.assistantRaw(res.Content, res.Reasoning, res.ToolCalls)); err != nil {
		return false, err
	}

	usage := parseUsage(res.Usage)
	rec := &UsageRecord{
		ID: newID("llm"), SessionID: a.sessionID, MessageID: msg.ID, Model: a.cfg.Model,
		ResponseID: res.ID, FinishReason: res.FinishReason, Usage: usage, RawUsage: res.Usage,
		LatencyMs: latency, CreatedAt: time.Now(),
	}
	if a.cfg.Billing != nil {
		rec.Cost = a.cfg.Billing.Cost(a.cfg.Model, usage)
	}
	if err := a.recordUsage(ctx, st, rec); err != nil {
		return false, err
	}
	st.res.FinishReason = res.FinishReason
	a.log.Debug("llm response", "session", a.sessionID, "finish_reason", res.FinishReason, "tool_calls", len(res.ToolCalls),
		"prompt_tokens", usage.PromptTokens, "cached_tokens", usage.CachedTokens, "completion_tokens", usage.CompletionTokens,
		"latency_ms", latency)
	calls := make([]RPCCall, 0, len(msg.ToolCalls))
	loaded := a.loadedDocsIn(history) // exactly what this request sent
	for i, tc := range msg.ToolCalls {
		c := a.newCall(ctx, msg, i, tc, loaded)
		if cutOff(res.FinishReason) && !json.Valid([]byte(tc.Function.Arguments)) {
			// The reply hit max_tokens while writing this call. A parse error
			// would only make the model resend the same call and be cut off
			// again; say what happened and how to avoid it.
			c.Status, c.Result = CallDone, rpcErrorResponse(c.RPCID, truncatedCallError(maxTokens))
		}
		if err := a.createCall(ctx, st, &c); err != nil {
			return false, err
		}
		calls = append(calls, c)
	}
	// Images go after every tool message of the turn to keep call/result pairing valid.
	if err := a.attachImages(ctx, st, calls); err != nil {
		return false, err
	}
	return len(msg.ToolCalls) > 0, nil
}

// recordInterrupted records an estimate of what LLM requests that broke off
// (Stop, a timeout, a dropped connection) cost upstream, which reports no
// usage for them. Without it a run stopped mid-reply is free, and stopping
// every reply just before it ends would make the whole conversation free.
// Each opened attempt is billed the estimated prompt — the part the previous
// call already sent as cache reads, the rest as input — and the output is
// estimated at 3 bytes a token. The record's finish_reason is "interrupted"
// and its raw usage says it is an estimate.
func (a *Agent) recordInterrupted(ctx context.Context, st *runState, spend streamSpend, est, prefix int, messageID string) error {
	if spend.opened == 0 || est <= 0 {
		return nil
	}
	if err := a.ownsLock(st); err != nil {
		return nil // recovered meanwhile: the new owner bills its own calls
	}
	if prefix > est {
		prefix = est
	}
	// One record per attempt. Only the last carries the message id and the
	// output (an attempt is retried only before any text), so the next
	// step's estimate (lastCallSize) reads one request, not the sum.
	for i := 1; i <= spend.opened; i++ {
		u := Usage{PromptTokens: int64(est), CachedTokens: int64(prefix)}
		id, outBytes := "", 0
		if i == spend.opened {
			id, outBytes = messageID, spend.outBytes
			u.CompletionTokens = int64((outBytes + 2) / 3)
		}
		u.TotalTokens = u.PromptTokens + u.CompletionTokens
		raw, _ := marshalJSON(struct {
			Estimated        bool  `json:"estimated"`
			PromptTokens     int64 `json:"prompt_tokens"`
			CachedTokens     int64 `json:"cached_tokens"`
			CompletionTokens int64 `json:"completion_tokens"`
			OutputBytes      int   `json:"output_bytes"`
		}{true, u.PromptTokens, u.CachedTokens, u.CompletionTokens, outBytes})
		rec := &UsageRecord{
			ID: newID("llm"), SessionID: a.sessionID, MessageID: id, Model: a.cfg.Model,
			FinishReason: FinishInterrupted, Usage: u, RawUsage: raw, CreatedAt: time.Now(),
		}
		if a.cfg.Billing != nil {
			rec.Cost = a.cfg.Billing.Cost(a.cfg.Model, u)
		}
		a.log.Info("llm request broke off, usage estimated", "session", a.sessionID,
			"prompt_tokens", u.PromptTokens, "cached_tokens", u.CachedTokens, "completion_tokens", u.CompletionTokens)
		if err := a.recordUsage(ctx, st, rec); err != nil {
			return err
		}
	}
	return nil
}

// FinishInterrupted is the finish_reason of a usage record estimated for an
// LLM request that broke off before reporting its usage.
const FinishInterrupted = "interrupted"

// cutOff reports a reply that stopped at the output limit ("length" in the
// OpenAI format, "max_tokens" from relays that pass Anthropic's through).
func cutOff(finishReason string) bool {
	return finishReason == "length" || finishReason == "max_tokens"
}

func truncatedCallError(maxTokens int) *RPCError {
	return &RPCError{Code: CodeParseError, Message: fmt.Sprintf(
		"your reply reached the output limit (%d tokens, reasoning included) while writing this call, so its arguments were cut off and nothing ran. "+
			"Do not resend it whole: it would be cut off again. Split the work into smaller calls that each write a part, and keep your reasoning before each call short.", maxTokens)}
}

// replayable drops messages that must not be sent to the model: interrupted
// assistant messages with no text, and anything not final.
func replayable(ms []Message) []Message {
	out := ms[:0:0]
	for _, m := range ms {
		switch {
		case m.Status == MessageInterrupted && m.Content == "":
		case m.Status == MessageExcluded:
		case !m.Status.Final():
		default:
			out = append(out, m)
		}
	}
	return out
}

// recordUsage stores one LLM call's usage and adds it to the run totals.
func (a *Agent) recordUsage(ctx context.Context, st *runState, rec *UsageRecord) error {
	if err := insertUsage(st.db, a.store, rec); err != nil {
		return err
	}
	st.res.Usage.add(rec.Usage)
	st.res.Cost.add(rec.Cost)
	if a.cfg.OnUsage != nil {
		a.callback("OnUsage", func() { a.cfg.OnUsage(ctx, *rec) })
	}
	return nil
}

var ephemeral = json.RawMessage(`{"type":"ephemeral"}`)

// withCacheControl returns the request messages with Anthropic-style cache
// breakpoints on the system prompt and on the last message (the rolling
// breakpoint). Only the request copy changes; stored messages stay as they are.
func withCacheControl(msgs []json.RawMessage) []json.RawMessage {
	out := append([]json.RawMessage(nil), msgs...)
	if len(out) > 0 {
		out[0] = addCacheControl(out[0])
	}
	if len(out) > 1 {
		out[len(out)-1] = addCacheControl(out[len(out)-1])
	}
	return out
}

// addCacheControl marks the last content part of a message, turning string
// content into a single text part. Messages without text content (e.g. an
// assistant turn that only calls tools) are returned unchanged.
func addCacheControl(raw json.RawMessage) json.RawMessage {
	var m map[string]json.RawMessage
	if json.Unmarshal(raw, &m) != nil {
		return raw
	}
	var parts []map[string]json.RawMessage
	var s string
	switch {
	case json.Unmarshal(m["content"], &s) == nil && s != "":
		text, _ := marshalJSON(s)
		parts = []map[string]json.RawMessage{{"type": json.RawMessage(`"text"`), "text": text}}
	case json.Unmarshal(m["content"], &parts) == nil && len(parts) > 0:
	default:
		return raw
	}
	parts[len(parts)-1]["cache_control"] = ephemeral
	content, err := marshalJSON(parts)
	if err != nil {
		return raw
	}
	m["content"] = content
	out, err := marshalJSON(m)
	if err != nil {
		return raw
	}
	return out
}
