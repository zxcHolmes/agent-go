package agent

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"
)

// monitor keeps the session locks of a Client's runs fresh and notices
// cross-process stops and lost locks. It works in batches: runs sharing the
// same intervals form a group served by one goroutine, which heartbeats all
// of them in one UPDATE and polls all of them in one SELECT. An idle process
// with many runs thus costs one read per StopPollInterval, not one per run.
type monitor struct {
	store  Store
	log    *slog.Logger
	mu     sync.Mutex
	groups map[monitorKey]*monitorGroup
}

type monitorKey struct{ beat, poll time.Duration }

type monitorGroup struct {
	runs map[*watchedRun]struct{}
	quit chan struct{}
}

type watchedRun struct {
	sessionID, runID string
	cancel           context.CancelCauseFunc
	log              *slog.Logger
	cancelled        bool // stop or lock loss already reported
}

// monitorBatch caps the ids per statement (two placeholders per run in the
// heartbeat), well under every dialect's parameter limit.
const monitorBatch = 200

// watch registers a run: its lock is refreshed every StaleAfter/4 (at most
// 5s) and, every StopPollInterval, the run is cancelled with ErrStopped if
// the session turned "stopping" or with ErrLockLost if another run owns it.
// The returned func unregisters it; once it returns the run is never
// cancelled by the monitor.
func (m *monitor) watch(a *Agent, st *runState) func() {
	beat := a.cfg.StaleAfter / 4
	if beat > 5*time.Second {
		beat = 5 * time.Second
	}
	k := monitorKey{beat: beat, poll: a.cfg.StopPollInterval} // poll <0: no polling
	w := &watchedRun{sessionID: a.sessionID, runID: st.runID, cancel: st.cancel, log: a.log}

	m.mu.Lock()
	if m.groups == nil {
		m.groups = make(map[monitorKey]*monitorGroup)
	}
	g := m.groups[k]
	if g == nil {
		g = &monitorGroup{runs: make(map[*watchedRun]struct{}), quit: make(chan struct{})}
		m.groups[k] = g
		go m.serve(k, g)
	}
	g.runs[w] = struct{}{}
	m.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			m.mu.Lock()
			defer m.mu.Unlock()
			delete(g.runs, w)
			if len(g.runs) == 0 {
				delete(m.groups, k)
				close(g.quit)
			}
		})
	}
}

func (m *monitor) serve(k monitorKey, g *monitorGroup) {
	tick := k.beat
	if k.poll > 0 && k.poll < tick {
		tick = k.poll
	}
	if tick < 50*time.Millisecond {
		tick = 50 * time.Millisecond
	}
	t := time.NewTicker(tick)
	defer t.Stop()
	lastBeat, lastPoll := time.Now(), time.Now()
	for {
		select {
		case <-g.quit:
			return
		case <-t.C:
		}
		beat := time.Since(lastBeat) >= k.beat
		poll := k.poll > 0 && time.Since(lastPoll) >= k.poll
		if !beat && !poll {
			continue
		}
		m.mu.Lock()
		runs := make([]*watchedRun, 0, len(g.runs))
		for w := range g.runs {
			runs = append(runs, w)
		}
		m.mu.Unlock()
		for len(runs) > 0 {
			n := min(len(runs), monitorBatch)
			if beat {
				m.beat(runs[:n])
			}
			if poll {
				m.poll(g, runs[:n])
			}
			runs = runs[n:]
		}
		if beat {
			lastBeat = time.Now()
		}
		if poll {
			lastPoll = time.Now()
		}
	}
}

// beat refreshes the locks still held by runs. Run ids are unique, so
// matching both lists is the same as matching each (id, run_id) pair, and
// the primary key serves the lookup. A run that ended or lost its lock no
// longer matches.
func (m *monitor) beat(runs []*watchedRun) {
	ph := placeholders(len(runs))
	args := make([]any, 0, 1+2*len(runs))
	args = append(args, nowMillis())
	for _, w := range runs {
		args = append(args, w.sessionID)
	}
	for _, w := range runs {
		args = append(args, w.runID)
	}
	if _, err := m.store.Exec(context.Background(),
		"UPDATE agent_sessions SET updated_at = ? WHERE id IN ("+ph+") AND run_id IN ("+ph+")", args...); err != nil {
		m.log.Warn("heartbeat failed", "runs", len(runs), "error", err)
	}
}

// poll reads the sessions of runs and cancels the runs that were stopped or
// lost their lock. Runs unregistered meanwhile are left alone.
func (m *monitor) poll(g *monitorGroup, runs []*watchedRun) {
	args := make([]any, len(runs))
	for i, w := range runs {
		args[i] = w.sessionID
	}
	rows, err := m.store.Query(context.Background(),
		"SELECT id, run_id, status FROM agent_sessions WHERE id IN ("+placeholders(len(runs))+")", args...)
	if err != nil {
		m.log.Warn("stop poll failed", "runs", len(runs), "error", err)
		return
	}
	type lock struct {
		runID  string
		status Status
	}
	locks := make(map[string]lock, len(runs))
	for rows.Next() {
		var id, runID, status string
		if err := rows.Scan(&id, &runID, &status); err != nil {
			rows.Close()
			m.log.Warn("stop poll failed", "runs", len(runs), "error", err)
			return
		}
		locks[id] = lock{runID, Status(status)}
	}
	err = rows.Err()
	rows.Close()
	if err != nil {
		m.log.Warn("stop poll failed", "runs", len(runs), "error", err)
		return
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, w := range runs {
		if _, ok := g.runs[w]; !ok || w.cancelled {
			continue
		}
		l, ok := locks[w.sessionID]
		switch {
		case !ok:
			w.log.Warn("stop poll failed", "session", w.sessionID, "error", ErrSessionNotFound)
		case l.runID != w.runID:
			w.log.Warn("session taken over by another run; abandoning this one", "session", w.sessionID, "run", w.runID)
			w.cancelled = true
			w.cancel(ErrLockLost)
		case l.status == StatusStopping:
			w.cancelled = true
			w.cancel(ErrStopped)
		}
	}
}

func placeholders(n int) string {
	return strings.TrimSuffix(strings.Repeat("?, ", n), ", ")
}
