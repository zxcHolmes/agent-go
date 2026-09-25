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
}

// run acquires the session and drives it. With loop=false it only heals the
// session and stops it (used by Stop).
func (a *Agent) run(ctx context.Context, from []Status, prepare func(context.Context, *runState) error, loop bool) (*RunResult, error) {
	runID, err := a.acquire(ctx, from)
	if err != nil {
		return nil, err
	}
	runCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	lr := &localRun{runID: runID, cancel: cancel}
	running.Store(a.sessionID, lr)
	defer running.CompareAndDelete(a.sessionID, lr)

	st := &runState{runID: runID, db: context.WithoutCancel(ctx), cancel: cancel, res: &RunResult{SessionID: a.sessionID}}
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

// heartbeat keeps the session lock fresh while the run is alive and watches
// for Stop from other processes (status "stopping"), so a stop interrupts even
// the middle of a stream or a long RPC call within ~500ms. The returned func
// stops it and waits for it to exit.
func (a *Agent) heartbeat(ctx context.Context, st *runState) func() {
	beatEvery := a.cfg.StaleAfter / 4
	if beatEvery > 5*time.Second {
		beatEvery = 5 * time.Second
	}
	pollEvery := 500 * time.Millisecond
	if beatEvery < pollEvery {
		pollEvery = beatEvery
	}
	if pollEvery < 50*time.Millisecond {
		pollEvery = 50 * time.Millisecond
	}
	quit := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		t := time.NewTicker(pollEvery)
		defer t.Stop()
		lastBeat := time.Now()
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
				_, _ = a.store.Exec(st.db, "UPDATE agent_sessions SET updated_at = ? WHERE id = ? AND run_id = ?", nowMillis(), a.sessionID, st.runID)
			}
			owner, status, err := a.sessionLock(st.db)
			switch {
			case err != nil:
			case owner != st.runID:
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
	if err != nil && !errors.Is(err, ErrWaitingConfirmation) {
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
	return res, err
}
