package agent

import (
	"context"
	"errors"
	"sync"
	"time"
)

var running sync.Map

type localRun struct {
	runID  string
	cancel context.CancelCauseFunc
}

type runState struct {
	runID    string
	db       context.Context // not cancelled by Stop, so bookkeeping always completes
	cancel   context.CancelCauseFunc
	startSeq int64 // messages with a greater seq were produced by this run
	res      *RunResult
	start    time.Time
}

// run acquires the session and drives it. With loop=false it only heals the
// session and stops it (used by Stop).
//
// Once the session is acquired the run no longer follows the caller's ctx
// (only its values): a run is ended by Stop, not by the request that started
// it going away.
//
// A message can be enqueued after the loop's last look at the queue but
// before the session turns idle. So when a run completes, run looks at the
// queue again with the session idle and, if something is waiting, runs once
// more to answer it (unless another run got the session first). The results
// are merged into one.
func (a *Agent) run(ctx context.Context, from []Status, prepare func(context.Context, *runState) error, loop bool) (*RunResult, error) {
	res, err := a.runOnce(ctx, from, prepare, loop)
	ctx = context.WithoutCancel(ctx)
	for loop && err == nil && res.StopReason == StopCompleted {
		qs, qerr := queuedMessages(ctx, a.store, a.sessionID, queueWaiting)
		if qerr != nil || len(qs) == 0 {
			break
		}
		a.log.Debug("messages queued as the run ended; running again", "session", a.sessionID, "count", len(qs))
		next, nerr := a.runOnce(ctx, []Status{StatusIdle}, func(ctx context.Context, st *runState) error {
			if n, err := a.drainQueue(ctx, st); err != nil || n > 0 {
				return err
			}
			return errNothingQueued // withdrawn, or taken by another run
		}, true)
		if nerr != nil && next == nil {
			break // another run has the session (ErrBusy, ErrWaitingConfirmation, ...)
		}
		if errors.Is(nerr, errNothingQueued) {
			break
		}
		res.Messages = append(res.Messages, next.Messages...)
		res.Usage.add(next.Usage)
		res.Cost.add(next.Cost)
		res.Status, res.StopReason, res.FinishReason, res.PendingCalls = next.Status, next.StopReason, next.FinishReason, next.PendingCalls
		err = nerr
	}
	if res != nil && a.cfg.OnRunEnd != nil {
		a.callback("OnRunEnd", func() { a.cfg.OnRunEnd(ctx, *res, err) })
	}
	return res, err
}

// errNothingQueued ends a follow-up run that found nothing left to answer.
var errNothingQueued = errors.New("agent: nothing queued")

func (a *Agent) runOnce(ctx context.Context, from []Status, prepare func(context.Context, *runState) error, loop bool) (*RunResult, error) {
	runID, err := a.acquire(ctx, from)
	if err != nil {
		return nil, err
	}
	ctx = context.WithoutCancel(ctx)
	runCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	lr := &localRun{runID: runID, cancel: cancel}
	running.Store(a.sessionID, lr)
	defer running.CompareAndDelete(a.sessionID, lr)

	st := &runState{runID: runID, db: context.WithoutCancel(ctx), cancel: cancel, res: &RunResult{SessionID: a.sessionID}, start: time.Now()}
	a.log.Info("run started", "session", a.sessionID, "run", runID)
	hbDone := a.heartbeat(runCtx, st)
	defer hbDone()

	var reason StopReason
	err = a.heal(runCtx, st)
	if err == nil && prepare != nil {
		err = prepare(runCtx, st)
	}
	switch {
	case err != nil:
	case loop:
		reason, err = a.loop(runCtx, st)
	default:
		st.cancel(ErrStopped)
		err = ErrStopped
	}
	hbDone()
	return a.finish(runCtx, st, reason, err)
}

// heartbeat keeps the session lock fresh while the run is alive (a write
// every StaleAfter/4, at most 5s) and, every Config.StopPollInterval, reads
// the session row to notice a Stop issued by another process (status
// "stopping") or a lost lock, so either interrupts even a long stream or RPC
// call. The returned func stops it and waits for it to exit.
func (a *Agent) heartbeat(ctx context.Context, st *runState) func() {
	beatEvery := a.cfg.StaleAfter / 4
	if beatEvery > 5*time.Second {
		beatEvery = 5 * time.Second
	}
	poll := a.cfg.StopPollInterval // <0: no polling
	tick := beatEvery
	if poll > 0 && poll < tick {
		tick = poll
	}
	if tick < 50*time.Millisecond {
		tick = 50 * time.Millisecond
	}
	quit := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		t := time.NewTicker(tick)
		defer t.Stop()
		lastBeat, lastPoll := time.Now(), time.Now()
		for {
			select {
			case <-quit:
				return
			case <-ctx.Done():
				return
			case <-t.C:
			}
			if time.Since(lastBeat) >= beatEvery {
				lastBeat = time.Now()
				if _, err := a.store.Exec(st.db, "UPDATE agent_sessions SET updated_at = ? WHERE id = ? AND run_id = ?", nowMillis(), a.sessionID, st.runID); err != nil {
					a.log.Warn("heartbeat failed", "session", a.sessionID, "error", err)
				}
			}
			if poll <= 0 || time.Since(lastPoll) < poll {
				continue
			}
			lastPoll = time.Now()
			owner, status, err := a.sessionLock(st.db)
			switch {
			case err != nil:
				a.log.Warn("stop poll failed", "session", a.sessionID, "error", err)
			case owner != st.runID:
				a.log.Warn("session taken over by another run; abandoning this one", "session", a.sessionID, "run", st.runID)
				st.cancel(ErrLockLost)
				return
			case status == StatusStopping:
				st.cancel(ErrStopped)
				return
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() { close(quit) })
		<-exited
	}
}

// heal brings the session back to a consistent state before running: it
// repairs leftovers of a crashed run (for sessions taken over without Init)
// and records tool calls whose records or tool messages were never written.
func (a *Agent) heal(ctx context.Context, st *runState) error {
	if err := recoverSessionData(st.db, a.store, a.sessionID); err != nil {
		return err
	}
	last, err := latestMessages(st.db, a.store, a.sessionID, 1)
	if err != nil {
		return err
	}
	if len(last) > 0 {
		st.startSeq = last[0].Seq
	}
	return a.reconcileCalls(ctx, st)
}

// reconcileCalls makes sure every tool call of the latest assistant message
// has an RPC call record and a tool message (a crash between writes can leave
// either missing).
func (a *Agent) reconcileCalls(ctx context.Context, st *runState) error {
	msg, err := lastAssistant(st.db, a.store, a.sessionID)
	if err != nil || msg == nil || len(msg.ToolCalls) == 0 {
		return err
	}
	calls, err := callsForMessage(st.db, a.store, msg.ID)
	if err != nil {
		return err
	}
	byIndex := make(map[int]*RPCCall, len(calls))
	for i := range calls {
		byIndex[calls[i].Index] = &calls[i]
	}
	for i, tc := range msg.ToolCalls {
		c := byIndex[i]
		if c == nil {
			nc := a.newCall(msg, i, tc)
			if err := a.createCall(ctx, st, &nc); err != nil {
				return err
			}
			continue
		}
		if _, err := getMessage(st.db, a.store, a.sessionID, c.ResultMessageID); errors.Is(err, ErrMessageNotFound) {
			m := a.toolMessage(c)
			m.ID = c.ResultMessageID
			if err := a.insertMessage(ctx, st, &m); err != nil {
				return err
			}
		} else if err != nil {
			return err
		}
	}
	// Tool messages are complete now; add any view_image message still missing.
	if calls, err = callsForMessage(st.db, a.store, msg.ID); err != nil {
		return err
	}
	return a.attachImages(ctx, st, calls)
}

func (a *Agent) finish(runCtx context.Context, st *runState, reason StopReason, err error) (*RunResult, error) {
	res := st.res
	if errors.Is(err, ErrLockLost) || errors.Is(context.Cause(runCtx), ErrLockLost) {
		return res, ErrLockLost
	}
	interrupted := err != nil && runCtx.Err() != nil
	stopped := interrupted && errors.Is(context.Cause(runCtx), ErrStopped)
	if !stopped {
		// Stop may have landed just as the run ended on its own; honour it.
		if owner, status, lerr := a.sessionLock(st.db); lerr == nil && owner == st.runID && status == StatusStopping {
			stopped = true
		}
	}
	if stopped {
		// A requested stop is a normal outcome, not an error.
		err = nil
	}
	var bookErr error // bookkeeping failures while wrapping up
	if stopped || interrupted {
		open, oerr := openCalls(st.db, a.store, a.sessionID)
		if oerr == nil {
			if stopped {
				oerr = a.cancelCalls(runCtx, st, open, true, errStoppedNotRun)
			} else {
				oerr = a.cancelCalls(runCtx, st, open, false, &RPCError{Code: CodeCancelled, Message: "cancelled before this call ran: " + context.Cause(runCtx).Error()})
			}
		}
		bookErr = errors.Join(bookErr, oerr)
	}
	pending, perr := pendingCalls(st.db, a.store, a.sessionID)
	bookErr = errors.Join(bookErr, perr)
	status := StatusIdle
	if len(pending) > 0 {
		status = StatusWaitingConfirmation
	}
	lastErr := ""
	if err != nil && !errors.Is(err, ErrWaitingConfirmation) && !errors.Is(err, errNothingQueued) {
		lastErr = err.Error()
	}
	bookErr = errors.Join(bookErr, a.release(st, status, lastErr))
	msgs, merr := messagesAfterSeq(st.db, a.store, a.sessionID, st.startSeq)
	bookErr = errors.Join(bookErr, merr)
	if bookErr != nil {
		err = errors.Join(err, bookErr)
	}
	res.Status, res.PendingCalls, res.Messages = status, pending, msgs
	switch {
	case err != nil:
	case stopped:
		res.StopReason = StopStopped
	case status == StatusWaitingConfirmation:
		res.StopReason = StopWaitingConfirmation
	default:
		res.StopReason = reason
	}
	attrs := []any{"session", a.sessionID, "run", st.runID, "status", string(res.Status), "stop_reason", string(res.StopReason),
		"duration_ms", time.Since(st.start).Milliseconds(), "prompt_tokens", res.Usage.PromptTokens,
		"completion_tokens", res.Usage.CompletionTokens, "credits", res.Cost.Total}
	switch {
	case errors.Is(err, errNothingQueued):
		a.log.Debug("follow-up run found nothing queued", attrs...)
	case err != nil:
		a.log.Warn("run finished with error", append(attrs, "error", err)...)
	default:
		a.log.Info("run finished", attrs...)
	}
	return res, err
}
