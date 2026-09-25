package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"
)

// LLMCallInfo describes an LLM request about to be made (Config.BeforeLLMCall).
type LLMCallInfo struct {
	SessionID string
	Model     string
	// Purpose is "chat" for agent steps and "summary" for compaction.
	Purpose string
	// EstimatedPromptTokens is the SDK's estimate of the prompt size.
	EstimatedPromptTokens int
	// MaxTokens is the output budget sent with the request (0 = provider default).
	MaxTokens int
	// MaxCost is the worst case under Config.Billing: the whole prompt billed
	// as uncached input plus MaxTokens of output. Zero without Billing.
	MaxCost Cost
}

// levelHandler filters records below a minimum level before passing them on,
// so each agent can have its own level on a shared logger.
type levelHandler struct {
	min   slog.Level
	inner slog.Handler
}

func (h levelHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return l >= h.min && h.inner.Enabled(ctx, l)
}
func (h levelHandler) Handle(ctx context.Context, r slog.Record) error { return h.inner.Handle(ctx, r) }
func (h levelHandler) WithAttrs(as []slog.Attr) slog.Handler {
	return levelHandler{h.min, h.inner.WithAttrs(as)}
}
func (h levelHandler) WithGroup(n string) slog.Handler {
	return levelHandler{h.min, h.inner.WithGroup(n)}
}

func newLogger(cfg Config) *slog.Logger {
	base := cfg.Logger
	if base == nil {
		base = slog.Default()
	}
	return slog.New(levelHandler{min: cfg.LogLevel, inner: base.Handler()}).With("component", "agent-go")
}

// beforeLLMCall runs the Config.BeforeLLMCall hook.
// callback runs a user callback, recovering and logging a panic so a buggy
// callback cannot take the run (or the process) down.
func (a *Agent) callback(name string, fn func()) {
	defer func() {
		if r := recover(); r != nil {
			a.log.Error("callback panicked", "session", a.sessionID, "callback", name, "panic", fmt.Sprint(r))
		}
	}()
	fn()
}

// onToolCall reports a call event to Config.OnToolCall.
func (a *Agent) onToolCall(ctx context.Context, phase ToolCallPhase, c *RPCCall) {
	if a.cfg.OnToolCall == nil {
		return
	}
	ev := ToolCallEvent{Phase: phase, Call: *c}
	if phase == ToolCallEnd && !c.started.IsZero() {
		ev.Duration = time.Since(c.started)
	}
	a.callback("OnToolCall", func() { a.cfg.OnToolCall(ctx, ev) })
}

func (a *Agent) beforeLLMCall(ctx context.Context, purpose string, est, maxTokens int) error {
	if a.cfg.BeforeLLMCall == nil {
		return nil
	}
	info := LLMCallInfo{SessionID: a.sessionID, Model: a.cfg.Model, Purpose: purpose, EstimatedPromptTokens: est, MaxTokens: maxTokens}
	if a.cfg.Billing != nil {
		out := maxTokens
		if out <= 0 {
			out = a.cfg.MaxOutputTokens
		}
		info.MaxCost = a.cfg.Billing.Cost(a.cfg.Model, Usage{PromptTokens: int64(est), CompletionTokens: int64(out)})
	}
	var err error
	a.callback("BeforeLLMCall", func() {
		err = errors.New("agent: BeforeLLMCall panicked") // replaced unless it panics
		err = a.cfg.BeforeLLMCall(ctx, info)
	})
	if err != nil {
		a.log.Info("llm call rejected by BeforeLLMCall", "session", a.sessionID, "purpose", purpose, "error", err)
		return err
	}
	return nil
}
