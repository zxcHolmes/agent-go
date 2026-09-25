package agent

import (
	"context"
	"errors"
	"time"
)

// Stop asks the session to stop and returns without waiting for the run to
// end; poll the session status to see it become idle. It is idempotent and
// safe to call concurrently, from any process sharing the store.
//
//   - running: the session turns "stopping" and the run is interrupted (at
//     once in this process, within Config.StopPollInterval in another). A
//     streaming answer is cut off and keeps its partial text ("interrupted"),
//     the LLM request is cancelled, a running RPC call is abandoned without
//     waiting for its handler (the handler's ctx is cancelled and its eventual
//     result discarded), and every unfinished call, including calls awaiting
//     confirmation, gets a "stopped by user" error result. The run then
//     leaves the session idle.
//   - waiting_confirmation: pending calls get "stopped by user" before Stop
//     returns; the session is idle.
//   - stopping, idle: returns nil.
//
// A run whose process died (no heartbeat for Config.StaleAfter) is taken over
// and cleaned up by Stop itself, so the session is idle when it returns. A
// dead run that is not stale yet stays "stopping" until the next run or
// NewClient's recovery takes it over.
func (a *Agent) Stop(ctx context.Context) error {
	a.log.Info("stop requested", "session", a.sessionID)
	for {
		s, err := getSession(ctx, a.store, a.sessionID)
		if err != nil {
			return err
		}
		switch s.Status {
		case StatusIdle:
			return nil
		case StatusWaitingConfirmation:
			err := a.stopIdle(ctx)
			if errors.Is(err, ErrBusy) {
				continue // a run started meanwhile (e.g. Confirm): stop that one
			}
			if err != nil && !errors.Is(err, ErrNoPendingCalls) {
				return err
			}
			return nil
		case StatusRunning, StatusStopping:
			if time.Since(s.UpdatedAt) > a.cfg.StaleAfter {
				// The owning process is gone: take over and clean up here.
				err := a.stopIdle(ctx)
				if errors.Is(err, ErrBusy) {
					continue // someone else took it over first
				}
				if err != nil && !errors.Is(err, ErrNoPendingCalls) {
					return err
				}
				return nil
			}
			if s.Status == StatusStopping {
				return nil
			}
			// updated_at is left alone so a dead run still looks stale.
			if _, err := a.store.Exec(ctx, "UPDATE agent_sessions SET status = ? WHERE id = ? AND status = ?",
				string(StatusStopping), a.sessionID, string(StatusRunning)); err != nil {
				return err
			}
			if v, ok := running.Load(a.sessionID); ok {
				v.(*localRun).cancel(ErrStopped)
			}
			continue // re-read: the run may have ended or started waiting for confirmation
		}
		return nil
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
