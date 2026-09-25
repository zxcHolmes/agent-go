package agent

import (
	"context"
	"errors"
	"time"
)

// Stop stops the session and blocks until it is idle. It is idempotent and
// safe to call concurrently, from any process sharing the store.
//
//   - running: the session turns "stopping"; the run is interrupted at once. A
//     streaming answer is cut off and keeps its partial text ("interrupted"),
//     the LLM request is cancelled, a running RPC call is abandoned without
//     waiting for its handler (the handler's ctx is cancelled and its eventual
//     result discarded), and every unfinished call, including calls awaiting
//     confirmation, gets a "stopped by user" error result.
//   - waiting_confirmation: pending calls get "stopped by user".
//   - stopping: waits for the stop in progress.
//   - idle: returns nil immediately.
//
// A run whose process died (no heartbeat for Config.StaleAfter) is taken over
// and cleaned up by Stop itself. If ctx ends first, Stop returns its error;
// the session still finishes stopping on its own.
func (a *Agent) Stop(ctx context.Context) error {
	for {
		s, err := getSession(ctx, a.store, a.sessionID)
		if err != nil {
			return err
		}
		switch s.Status {
		case StatusIdle:
			return nil
		case StatusWaitingConfirmation:
			if err := a.stopIdle(ctx); err != nil && !errors.Is(err, ErrBusy) && !errors.Is(err, ErrNoPendingCalls) {
				return err
			}
			continue // re-read: someone may have raced us
		case StatusRunning, StatusStopping:
			if s.Status == StatusRunning {
				// updated_at is left alone so a dead run still looks stale.
				if _, err := a.store.Exec(ctx, "UPDATE agent_sessions SET status = ? WHERE id = ? AND status = ?",
					string(StatusStopping), a.sessionID, string(StatusRunning)); err != nil {
					return err
				}
			}
			if v, ok := running.Load(a.sessionID); ok {
				v.(*localRun).cancel(ErrStopped)
			}
			if time.Since(s.UpdatedAt) > a.cfg.StaleAfter {
				// The owning process is gone: take over and clean up here.
				if err := a.stopIdle(ctx); err != nil && !errors.Is(err, ErrBusy) && !errors.Is(err, ErrNoPendingCalls) {
					return err
				}
				continue
			}
		}
		t := time.NewTimer(20 * time.Millisecond)
		select {
		case <-ctx.Done():
			t.Stop()
			return context.Cause(ctx)
		case <-t.C:
		}
	}
}

// stopIdle takes the session (waiting for confirmation, or abandoned by a dead
// run), repairs it, cancels every unfinished call and leaves it idle.
func (a *Agent) stopIdle(ctx context.Context) error {
	_, err := a.run(ctx, []Status{StatusWaitingConfirmation}, nil, false)
	return err
}

var (
	errStoppedNotRun  = &RPCError{Code: CodeCancelled, Message: "stopped by user: this call was not executed"}
	errStoppedRunning = &RPCError{Code: CodeCancelled, Message: "stopped by user while this call was running; it may or may not have taken effect"}
)
