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

func (a *Agent) loop(ctx context.Context, st *runState) (StopReason, error) {
	llmCalls := 0
	for {
		if err := a.checkpoint(ctx, st); err != nil {
			return "", err
		}
		open, err := openCalls(st.db, a.store, a.sessionID)
		if err != nil {
			return "", err
		}
		if len(open) > 0 {
			waiting := false
			var runnable []*RPCCall
			for i := range open {
				switch open[i].Status {
				case CallQueued, CallApproved:
					runnable = append(runnable, &open[i])
				case CallAwaitingConfirmation:
					waiting = true
				}
			}
			if err := a.executeAll(ctx, st, runnable); err != nil {
				return "", err
			}
			if waiting {
				return StopWaitingConfirmation, nil
			}
			continue
		}
		if llmCalls >= a.cfg.MaxSteps {
			return StopMaxSteps, nil
		}
		llmCalls++
		hasCalls, err := a.step(ctx, st)
		if err != nil {
			if healed, herr := a.dropUnloadableImages(ctx, st, err); herr != nil {
				return "", errors.Join(err, herr)
			} else if healed {
				llmCalls--
				continue // retry without the images the provider could not load
			}
			return "", err
		}
		if !hasCalls {
			// Messages queued while the model was answering still need an answer.
			if queued, err := a.hasQueued(st); err != nil {
				return "", err
			} else if queued {
				continue
			}
			return StopCompleted, nil
		}
	}
}

// executeAll runs the calls of one turn, in parallel up to
// Config.ToolConcurrency (0 = all at once, 1 = one after another). Results
// land in tool messages created up front in the model's order, so the order
// seen by the model does not depend on which call finishes first. It returns
// the first error after every started call has finished.
func (a *Agent) executeAll(ctx context.Context, st *runState, calls []*RPCCall) error {
	limit := a.cfg.ToolConcurrency
	if limit <= 0 || limit > len(calls) {
		limit = len(calls)
	}
	if limit <= 1 {
		for _, c := range calls {
			if err := a.execute(ctx, st, c); err != nil {
				return err
			}
		}
		return nil
	}
	sem := make(chan struct{}, limit)
	errs := make([]error, len(calls))
	var wg sync.WaitGroup
	for i, c := range calls {
		sem <- struct{}{}
		wg.Add(1)
		go func(i int, c *RPCCall) {
			defer func() { <-sem; wg.Done() }()
			errs[i] = a.execute(ctx, st, c)
		}(i, c)
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// execute runs one call. The tool message is marked "running" before the
// handler starts, so a crash mid-call is visible and recoverable. If the run
// is stopped meanwhile, the call is closed at once without waiting for the
// handler, whose late result is discarded.
func (a *Agent) execute(ctx context.Context, st *runState, c *RPCCall) error {
	if err := a.checkpoint(ctx, st); err != nil {
		return err
	}
	if err := setCallStatus(st.db, a.store, st.lease, c, CallRunning); err != nil {
		return err
	}
	a.notify(ctx, st, c.ResultMessageID)
	c.started = time.Now()
	a.onToolCall(ctx, ToolCallStart, c)
	timeout := a.cfg.RPCTimeout
	if m, ok := a.methods[c.Method]; ok && m.Timeout > 0 {
		timeout = m.Timeout
	}
	callCtx := ctx
	if timeout > 0 {
		var cancel context.CancelFunc
		callCtx, cancel = context.WithTimeout(ctx, timeout)
		defer cancel()
	}
	start := time.Now()
	done := make(chan json.RawMessage, 1)
	go func() { done <- a.invoke(callCtx, c) }()
	status := CallDone
	var result json.RawMessage
	select {
	case result = <-done:
	case <-callCtx.Done():
		select {
		case result = <-done: // finished at the same instant
		default:
		}
	}
	// A handler that honours ctx returns an error as soon as it is cancelled;
	// that error is a consequence of the stop or timeout, so report the stop
	// or timeout instead. A successful result is kept: the work did happen.
	if callCtx.Err() != nil && (result == nil || isErrorResponse(result)) {
		switch {
		case ctx.Err() == nil: // our timeout, not a stop
			result = rpcErrorResponse(c.RPCID, &RPCError{Code: CodeTimeout,
				Message: fmt.Sprintf("timed out after %s; the call may or may not have taken effect", timeout)})
			a.log.Info("rpc call timed out", "session", a.sessionID, "method", c.Method, "timeout", timeout)
		case errors.Is(context.Cause(ctx), ErrStopped):
			status, result = CallCancelled, rpcErrorResponse(c.RPCID, errStoppedRunning)
		default:
			status, result = CallCancelled, rpcErrorResponse(c.RPCID, &RPCError{Code: CodeCancelled,
				Message: "cancelled while this call was running (" + context.Cause(ctx).Error() + "); it may or may not have taken effect"})
		}
	}
	a.log.Debug("rpc call", "session", a.sessionID, "method", c.Method, "duration_ms", time.Since(start).Milliseconds(),
		"status", string(status), "result_bytes", len(result))
	if err := a.ownsLock(st); err != nil {
		return err // the session was recovered meanwhile; do not overwrite it
	}
	if err := a.completeCall(ctx, st, c, status, result); err != nil {
		return err
	}
	if status == CallCancelled {
		return context.Cause(ctx)
	}
	return nil
}

func (a *Agent) newCall(m *Message, i int, tc ToolCall) RPCCall {
	now := time.Now()
	c := RPCCall{
		ID: newID("call"), SessionID: a.sessionID, MessageID: m.ID, ToolCallID: tc.ID, Index: i,
		Method: tc.Function.Name, Status: CallQueued, ResultMessageID: newID("msg"), CreatedAt: now, UpdatedAt: now,
	}
	fallbackID, _ := marshalJSON(tc.ID)
	if a.hasDocs() && tc.Function.Name == ReadDocTool {
		a.newReadDocCall(&c, tc.Function.Arguments)
		return c
	}
	if a.cfg.ViewImage && tc.Function.Name == ViewImageTool {
		a.newViewImageCall(&c, tc.Function.Arguments)
		return c
	}
	if tc.Function.Name != a.cfg.ToolName || len(a.methods) == 0 {
		c.Status = CallDone
		c.Result = rpcErrorResponse(nil, &RPCError{Code: CodeMethodNotFound, Message: fmt.Sprintf("unknown tool %q; available tools: %s", tc.Function.Name, strings.Join(a.toolNames(), ", "))})
		return c
	}
	req, rerr := parseRPCRequest(tc.Function.Arguments)
	c.Method, c.Params, c.RPCID = req.Method, req.Params, req.ID
	if len(c.RPCID) == 0 || string(c.RPCID) == "null" {
		c.RPCID = fallbackID
	}
	if rerr != nil {
		c.Status, c.Result = CallDone, rpcErrorResponse(c.RPCID, rerr)
		return c
	}
	m2, ok := a.methods[c.Method]
	if !ok {
		c.Status, c.Result = CallDone, rpcErrorResponse(c.RPCID, a.methodNotFound(c.Method))
		return c
	}
	if m2.RequireConfirm {
		c.RequireConfirm, c.Status = true, CallAwaitingConfirmation
	}
	return c
}

func (a *Agent) toolNames() []string {
	var names []string
	if len(a.methods) > 0 {
		names = append(names, a.cfg.ToolName)
	}
	if a.hasDocs() {
		names = append(names, ReadDocTool)
	}
	if a.cfg.ViewImage {
		names = append(names, ViewImageTool)
	}
	if len(names) == 0 {
		return []string{"(none)"}
	}
	return names
}

// toolMessage builds the tool message mirroring a call's current state.
func (a *Agent) toolMessage(c *RPCCall) Message {
	status, content := MessagePending, ""
	if c.Status == CallRunning {
		status = MessageRunning
	}
	if c.Status.finished() {
		status, content = MessageDone, toolContent(c.Result)
	}
	return newMessage(a.sessionID, toolMessageRaw(c.ToolCallID, content), status)
}

// createCall writes a call record and its tool message.
func (a *Agent) createCall(ctx context.Context, st *runState, c *RPCCall) error {
	if err := insertCall(st.db, a.store, c); err != nil {
		return err
	}
	m := a.toolMessage(c)
	m.ID = c.ResultMessageID
	if err := a.insertMessage(ctx, st, &m); err != nil {
		return err
	}
	if c.Status.finished() { // answered at creation, never runs
		a.onToolCall(ctx, ToolCallEnd, c)
	}
	return nil
}

func (a *Agent) completeCall(ctx context.Context, st *runState, c *RPCCall, status CallStatus, result json.RawMessage) error {
	if err := completeCall(st.db, a.store, st.lease, c, status, result); err != nil {
		return err
	}
	a.notify(ctx, st, c.ResultMessageID)
	a.onToolCall(ctx, ToolCallEnd, c)
	return nil
}

// cancelCalls closes calls that never started with rerr. Calls awaiting
// confirmation are closed too when includeAwaiting is set (Stop).
func (a *Agent) cancelCalls(ctx context.Context, st *runState, open []RPCCall, includeAwaiting bool, rerr *RPCError) error {
	for i := range open {
		c := &open[i]
		switch c.Status {
		case CallQueued, CallApproved:
		case CallAwaitingConfirmation:
			if !includeAwaiting {
				continue
			}
		default:
			continue
		}
		if err := a.completeCall(ctx, st, c, CallCancelled, rpcErrorResponse(c.RPCID, rerr)); err != nil {
			return err
		}
	}
	return nil
}

func (a *Agent) insertMessage(ctx context.Context, st *runState, m *Message) error {
	if err := insertMessage(st.db, a.store, m); err != nil {
		return err
	}
	if a.cfg.OnMessage != nil {
		a.callback("OnMessage", func() { a.cfg.OnMessage(ctx, *m) })
	}
	return nil
}

func (a *Agent) updateMessage(ctx context.Context, st *runState, m *Message) error {
	if err := updateMessage(st.db, a.store, st.lease, m); err != nil {
		return err
	}
	if a.cfg.OnMessage != nil {
		a.callback("OnMessage", func() { a.cfg.OnMessage(ctx, *m) })
	}
	return nil
}

// notify reloads a message changed in the store and passes it to OnMessage.
func (a *Agent) notify(ctx context.Context, st *runState, messageID string) {
	if a.cfg.OnMessage == nil {
		return
	}
	if m, err := getMessage(st.db, a.store, a.sessionID, messageID); err == nil {
		a.callback("OnMessage", func() { a.cfg.OnMessage(ctx, *m) })
	}
}

// isErrorResponse reports whether a JSON-RPC response carries an error.
func isErrorResponse(resp json.RawMessage) bool {
	var r struct {
		Error json.RawMessage `json:"error"`
	}
	return json.Unmarshal(resp, &r) == nil && len(r.Error) > 0 && string(r.Error) != "null"
}
