package agent

import (
	"context"
	"encoding/json"
	"errors"
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
	history, maxTokens, err := a.prepareContext(ctx, st)
	if err != nil {
		return false, err
	}
	msgs := make([]json.RawMessage, 0, len(history)+1)
	if a.system != nil {
		msgs = append(msgs, a.system)
	}
	for _, m := range history {
		msgs = append(msgs, m.Raw)
	}
	req := chatRequest{Model: a.cfg.Model, Messages: msgs, Tools: a.tools}
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
			a.cfg.OnStream(ctx, msg.ID, dc, dr)
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
	res, err := a.llm.stream(ctx, req, a.cfg.ExtraBody, onDelta)
	endStream()
	mu.Lock() // pairs with a timer callback that may have just run
	defer mu.Unlock()
	if err != nil {
		if msg != nil {
			// Keep what was generated; partial tool calls are dropped.
			if ferr := flush(MessageInterrupted, partial()); ferr != nil {
				err = errors.Join(err, ferr)
			}
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
	calls := make([]RPCCall, 0, len(msg.ToolCalls))
	for i, tc := range msg.ToolCalls {
		c := a.newCall(msg, i, tc)
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
		a.cfg.OnUsage(ctx, *rec)
	}
	return nil
}
