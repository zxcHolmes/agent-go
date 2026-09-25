package examples

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

type runEnd struct {
	res agent.RunResult
	err error
}

// recorder collects OnRunEnd and OnToolCall events.
type recorder struct {
	mu    sync.Mutex
	runs  []runEnd
	calls []string // "phase method status code"
	durs  map[string]time.Duration
}

func (r *recorder) install(cfg *agent.Config) {
	r.durs = map[string]time.Duration{}
	cfg.OnRunEnd = func(ctx context.Context, res agent.RunResult, err error) {
		r.mu.Lock()
		r.runs = append(r.runs, runEnd{res, err})
		r.mu.Unlock()
	}
	cfg.OnToolCall = func(ctx context.Context, ev agent.ToolCallEvent) {
		code := ""
		if ev.Phase == agent.ToolCallEnd {
			if i := strings.Index(string(ev.Call.Result), `"code":`); i >= 0 {
				code = strings.SplitN(string(ev.Call.Result)[i+7:], ",", 2)[0]
			} else {
				code = "ok"
			}
		}
		r.mu.Lock()
		r.calls = append(r.calls, strings.TrimSpace(fmt.Sprintf("%s %s %s %s", ev.Phase, ev.Call.Method, ev.Call.Status, code)))
		if ev.Phase == agent.ToolCallEnd {
			r.durs[ev.Call.Method] = ev.Duration
		}
		r.mu.Unlock()
	}
}

func (r *recorder) takeRuns() []runEnd {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.runs
	r.runs = nil
	return out
}

func (r *recorder) takeCalls() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := strings.Join(r.calls, " | ")
	r.calls = nil
	return out
}

// OnRunEnd fires exactly once per run, whatever the outcome.
func TestOnRunEnd(t *testing.T) {
	e := setup(t, agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
		return "ok", nil
	}, agent.MethodDoc{RequireConfirm: true}))
	var rec recorder
	rec.install(&e.cfg)
	ctx := context.Background()
	a := e.newAgent(t, "")
	one := func(what string) runEnd {
		t.Helper()
		runs := rec.takeRuns()
		if len(runs) != 1 {
			t.Fatalf("%s: %d OnRunEnd calls", what, len(runs))
		}
		return runs[0]
	}

	e.llm.push(toolCall("c1", "get_order", `{"order_id":"O"}`), text("done"))
	res, _ := a.Chat(ctx, "hi")
	if r := one("completed"); r.err != nil || r.res.StopReason != agent.StopCompleted || r.res.Reply() != "done" || r.res.Usage != res.Usage {
		t.Fatalf("%+v", r)
	}

	e.llm.push(toolCall("c2", "refund", `{"order_id":"O"}`))
	_, _ = a.Chat(ctx, "refund")
	if r := one("waiting"); r.res.Status != agent.StatusWaitingConfirmation || len(r.res.PendingCalls) != 1 {
		t.Fatalf("%+v", r)
	}
	if _, err := a.Chat(ctx, "busy?"); !errors.Is(err, agent.ErrWaitingConfirmation) {
		t.Fatal(err)
	}
	if runs := rec.takeRuns(); len(runs) != 0 {
		t.Fatal("OnRunEnd for a run that never started")
	}
	// Stop cleaning up a session with no live run is reported too.
	if err := a.Stop(ctx); err != nil {
		t.Fatal(err)
	}
	if r := one("stop while waiting"); r.res.StopReason != agent.StopStopped || r.res.Status != agent.StatusIdle {
		t.Fatalf("%+v", r)
	}

	e.llm.pushReply(reply{msg: text("abcdef"), hangAfter: 1})
	done := make(chan struct{})
	go func() { _, _ = a.Chat(ctx, "long"); close(done) }()
	<-e.llm.hung
	_ = a.Stop(ctx)
	<-done
	if r := one("stopped"); r.err != nil || r.res.StopReason != agent.StopStopped {
		t.Fatalf("%+v", r)
	}

	e.llm.pushReply(reply{httpErr: 400, body: `{"error":{"message":"bad"}}`})
	_, err := a.Chat(ctx, "fail")
	if r := one("error"); r.err == nil || r.err.Error() != err.Error() {
		t.Fatalf("%+v vs %v", r, err)
	}
}

// OnToolCall reports start/end of executed calls and end of calls answered
// without running, with their final status and error code.
func TestOnToolCall(t *testing.T) {
	e := setup(t,
		agent.NewMethod("refund", func(ctx context.Context, c *agent.Call, p orderParams) (string, error) {
			return "ok", nil
		}, agent.MethodDoc{RequireConfirm: true}),
		agent.Method{Name: "slow", Timeout: 50 * time.Millisecond, Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			<-ctx.Done()
			return nil, ctx.Err()
		}},
		agent.Method{Name: "nap", Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			time.Sleep(20 * time.Millisecond)
			return "rested", nil
		}},
	)
	e.cfg.ToolConcurrency = 1
	var rec recorder
	rec.install(&e.cfg)
	ctx := context.Background()
	a := e.newAgent(t, "")

	e.llm.push(`{"role":"assistant","content":null,"tool_calls":[
		{"id":"a","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"nap\",\"params\":{},\"id\":1}"}},
		{"id":"b","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"get_order\",\"params\":{},\"id\":2}"}},
		{"id":"c","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"nope\",\"params\":{},\"id\":3}"}},
		{"id":"d","type":"function","function":{"name":"json_rpc","arguments":"{\"method\":\"slow\",\"params\":{},\"id\":4}"}}]}`,
		text("done"))
	if _, err := a.Chat(ctx, "go"); err != nil {
		t.Fatal(err)
	}
	// Calls answered at creation (unknown method) come first, then execution in
	// order; invalid params are detected when the call runs.
	if got := rec.takeCalls(); got != "end nope done -32601 | start nap running | end nap done ok | start get_order running | end get_order done -32602 | start slow running | end slow done -32004" {
		t.Fatal(got)
	}
	if d := rec.durs["nap"]; d < 20*time.Millisecond {
		t.Fatalf("nap duration %v", d)
	}

	// Awaiting confirmation: no event until decided; a rejection ends it
	// without a start.
	e.llm.push(toolCall("r1", "refund", `{"order_id":"O"}`), text("ok"))
	res, _ := a.Chat(ctx, "refund")
	if got := rec.takeCalls(); got != "" {
		t.Fatal(got)
	}
	if _, err := a.Confirm(ctx, agent.Reject(res.PendingCalls[0].ID, "no")); err != nil {
		t.Fatal(err)
	}
	if got := rec.takeCalls(); got != "end refund rejected -32001" {
		t.Fatal(got)
	}
}

// A panicking callback is logged and ignored; a panicking BeforeLLMCall
// rejects the request like an error.
func TestCallbackPanicsAreRecovered(t *testing.T) {
	e := setup(t)
	boom := func() { panic("boom") }
	e.cfg.OnStream = func(context.Context, string, string, string) { boom() }
	e.cfg.OnMessage = func(context.Context, agent.Message) { boom() }
	e.cfg.OnUsage = func(context.Context, agent.UsageRecord) { boom() }
	e.cfg.OnToolCall = func(context.Context, agent.ToolCallEvent) { boom() }
	e.cfg.OnRunEnd = func(context.Context, agent.RunResult, error) { boom() }
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("c1", "get_order", `{"order_id":"O"}`), text("fine"))
	res, err := a.Chat(ctx, "hi")
	if err != nil || res.Reply() != "fine" {
		t.Fatalf("%+v %v", res, err)
	}
	checkHistory(t, e, a.SessionID())

	e.cfg.BeforeLLMCall = func(context.Context, agent.LLMCallInfo) error { panic("boom") }
	a = e.newAgent(t, a.SessionID())
	if _, err := a.Chat(ctx, "again"); err == nil || !strings.Contains(err.Error(), "BeforeLLMCall panicked") {
		t.Fatal(err)
	}
	checkHistory(t, e, a.SessionID())
}
