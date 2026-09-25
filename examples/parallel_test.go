package examples

import (
	"context"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
)

func threeSlowCalls() string {
	return `{"role":"assistant","content":null,"tool_calls":[
		{"id":"a","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"slow\",\"params\":{\"n\":1},\"id\":1}"}},
		{"id":"b","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"slow\",\"params\":{\"n\":2},\"id\":2}"}},
		{"id":"c","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"slow\",\"params\":{\"n\":3},\"id\":3}"}}]}`
}

func slowMethod(active, peak *atomic.Int32, d time.Duration) agent.Method {
	return agent.Method{Name: "slow", Handler: func(ctx context.Context, c *agent.Call) (any, error) {
		n := active.Add(1)
		for {
			p := peak.Load()
			if n <= p || peak.CompareAndSwap(p, n) {
				break
			}
		}
		defer active.Add(-1)
		select {
		case <-time.After(d):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		var p struct{ N int }
		_ = c.Bind(&p)
		return p.N, nil
	}}
}

func TestParallelToolCalls(t *testing.T) {
	for _, tc := range []struct {
		concurrency int
		wantPeak    int32
		minTime     time.Duration
		maxTime     time.Duration
	}{
		{0, 3, 0, 550 * time.Millisecond},                       // default: all at once
		{2, 2, 550 * time.Millisecond, 950 * time.Millisecond},  // at most two
		{1, 1, 850 * time.Millisecond, 2000 * time.Millisecond}, // sequential
	} {
		var active, peak atomic.Int32
		e := setup(t, slowMethod(&active, &peak, 300*time.Millisecond))
		e.cfg.ToolConcurrency = tc.concurrency
		a := e.newAgent(t, "")
		e.llm.push(threeSlowCalls(), text("done"))
		start := time.Now()
		res, err := a.Chat(context.Background(), "go")
		took := time.Since(start)
		if err != nil {
			t.Fatal(err)
		}
		if peak.Load() != tc.wantPeak || took < tc.minTime || took > tc.maxTime {
			t.Fatalf("concurrency=%d: peak=%d took=%v", tc.concurrency, peak.Load(), took)
		}
		// Results stay in the model's call order.
		var got []string
		for _, m := range res.Messages {
			if m.Role == "tool" {
				got = append(got, m.ToolCallID+"="+m.Content[strings.Index(m.Content, `"result"`):])
			}
		}
		if strings.Join(got, " ") != `a="result":1} b="result":2} c="result":3}` {
			t.Fatalf("order: %v", got)
		}
	}
}

func TestStopDuringParallelCalls(t *testing.T) {
	var active, peak atomic.Int32
	e := setup(t, slowMethod(&active, &peak, 10*time.Second))
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(threeSlowCalls())
	done := make(chan *agent.RunResult)
	go func() {
		res, _ := a.Chat(ctx, "go")
		done <- res
	}()
	waitFor(t, "all three running", func() bool { return active.Load() == 3 })
	start := time.Now()
	if err := a.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	res := <-done
	if time.Since(start) > time.Second || res.StopReason != agent.StopStopped {
		t.Fatalf("%v %s", time.Since(start), res.StopReason)
	}
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	for _, m := range ms {
		if m.Role == "tool" && !strings.Contains(m.Content, "stopped by user while this call was running") {
			t.Fatalf("%s: %s", m.ToolCallID, m.Content)
		}
	}
}

func TestDeleteSession(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{Name: "block", Handler: func(ctx context.Context, c *agent.Call) (any, error) {
		close(started)
		<-release
		return "ok", nil
	}})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "block", `{}`), text("done"))
	_, _ = a.Enqueue(ctx, "queued")
	go func() { _, _ = a.Chat(ctx, "go") }()
	<-started
	if err := e.client.DeleteSession(ctx, a.SessionID()); !errors.Is(err, agent.ErrBusy) {
		t.Fatalf("deleting a running session: %v", err)
	}
	close(release)
	waitFor(t, "idle", func() bool { st, _ := a.Status(ctx); return st == agent.StatusIdle })

	sid := a.SessionID()
	if err := e.client.DeleteSession(ctx, sid); err != nil {
		t.Fatal(err)
	}
	if _, err := e.client.Session(ctx, sid); !errors.Is(err, agent.ErrSessionNotFound) {
		t.Fatal(err)
	}
	if ms, _ := e.client.LatestMessages(ctx, sid, 10); len(ms) != 0 {
		t.Fatalf("%d messages left", len(ms))
	}
	// Usage survives and can still be settled.
	bill, err := e.client.SettleUsage(ctx, sid, func(context.Context, *agent.Bill) error { return nil })
	if err != nil || bill == nil || bill.Summary.Calls != 2 {
		t.Fatalf("%+v %v", bill, err)
	}
	if err := e.client.DeleteSession(ctx, sid); !errors.Is(err, agent.ErrSessionNotFound) {
		t.Fatalf("deleting twice: %v", err)
	}
	if _, err := e.newAgentErr(sid); !errors.Is(err, agent.ErrSessionNotFound) {
		t.Fatalf("agent on a deleted session: %v", err)
	}
}

func (e *env) newAgentErr(sid string) (*agent.Agent, error) {
	return e.client.Agent(context.Background(), sid, agent.AgentOptions{})
}
