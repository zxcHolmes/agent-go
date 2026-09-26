package examples

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

// simulateRestart creates a new client on the same database, as a restarted process would.
func simulateRestart(t *testing.T, e *env) {
	t.Helper()
	if _, err := agent.NewClient(context.Background(), e.cfg); err != nil {
		t.Fatal(err)
	}
}

func TestCrashDuringToolCall(t *testing.T) {
	started := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			<-ctx.Done() // the "crashed" run never finishes on its own
			return nil, ctx.Err()
		},
	})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("a", "slow", `{}`))
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "go")
		done <- err
	}()
	<-started

	simulateRestart(t, e)
	s, _ := a.Session(ctx)
	if s.Status != agent.StatusIdle || !strings.Contains(s.LastError, "recovered") {
		t.Fatalf("session after recovery: %+v", s)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if statuses(all) != "done,done,done" || !strings.Contains(all[2].Content, `"code":-32003`) {
		t.Fatalf("%s %s", statuses(all), all[2].Content)
	}
	// The zombie run notices it lost the session and does not overwrite anything.
	if err := <-done; !errors.Is(err, agent.ErrLockLost) {
		t.Fatalf("want ErrLockLost, got %v", err)
	}
	all2, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if all2[2].Content != all[2].Content {
		t.Fatal("zombie run overwrote the recovered result")
	}

	e.llm.push(text("That call failed, sorry."))
	res, err := a.Chat(ctx, "what happened?")
	if err != nil || res.Reply() != "That call failed, sorry." {
		t.Fatal(res, err)
	}
}

func TestCrashDuringStream(t *testing.T) {
	e := setup(t)
	e.cfg.StreamFlushInterval = time.Millisecond
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("partial answer here"), hangAfter: 2})
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "hi")
		done <- err
	}()
	<-e.llm.hung
	waitFor(t, "partial flush", func() bool {
		ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 1)
		return len(ms) == 1 && ms[0].Content == "partia"
	})
	if st, _ := a.Status(ctx); st != agent.StatusRunning {
		t.Fatal(st)
	}

	simulateRestart(t, e)
	if st, _ := a.Status(ctx); st != agent.StatusIdle {
		t.Fatalf("status after restart: %s", st)
	}
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 1)
	if ms[0].Status != agent.MessageInterrupted || ms[0].Content != "partia" {
		t.Fatalf("%+v", ms[0])
	}
	if err := <-done; !errors.Is(err, agent.ErrLockLost) {
		t.Fatalf("want ErrLockLost, got %v", err)
	}

	e.llm.push(text("continuing"))
	if res, err := a.Chat(ctx, "go on"); err != nil || res.Reply() != "continuing" {
		t.Fatal(res, err)
	}
}

func TestRecoverStaleOnly(t *testing.T) {
	e := setup(t)
	e.cfg.StaleAfter = 10 * time.Second // heartbeat every ~2.5s
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("abcdef"), hangAfter: 1})
	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "hi")
		done <- err
	}()
	<-e.llm.hung
	// A live run on "another instance" must survive a restart that only recovers stale sessions.
	cfg := e.cfg
	cfg.RecoverStaleAfter = time.Minute
	if _, err := agent.NewClient(ctx, cfg); err != nil {
		t.Fatal(err)
	}
	if st, _ := a.Status(ctx); st != agent.StatusRunning {
		t.Fatalf("live run was reset: %s", st)
	}
	_ = a.Stop(ctx)
	if err := <-done; err != nil {
		t.Fatal(err)
	}
}

// takeoverStore hands the session to another run right before the first
// write of a finished RPC call, i.e. after the run's last ownership check.
type takeoverStore struct {
	agent.Store
	once sync.Once
}

func (s *takeoverStore) Exec(ctx context.Context, q string, args ...any) (int64, error) {
	if strings.HasPrefix(q, "UPDATE agent_rpc_calls") && len(args) > 0 && args[0] == string(agent.CallDone) {
		var err error
		s.once.Do(func() {
			_, err = s.Store.Exec(ctx, "UPDATE agent_sessions SET run_id = 'run_intruder'")
		})
		if err != nil {
			return 0, err
		}
	}
	return s.Store.Exec(ctx, q, args...)
}

// TestLostLockBeforeWrite: a run that loses the session between its ownership
// check and its write must not overwrite anything (the writes are fenced).
func TestLostLockBeforeWrite(t *testing.T) {
	e := setup(t)
	e.cfg.Store = &takeoverStore{Store: e.store}
	ctx := context.Background()
	client, err := agent.NewClient(ctx, e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, err := client.Agent(ctx, "", agent.AgentOptions{})
	if err != nil {
		t.Fatal(err)
	}
	e.llm.push(toolCall("a", "get_order", `{"order_id":"O1"}`))
	if _, err := a.Chat(ctx, "go"); !errors.Is(err, agent.ErrLockLost) {
		t.Fatalf("want ErrLockLost, got %v", err)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if statuses(all) != "done,done,running" || strings.Contains(all[2].Content, "O1") {
		t.Fatalf("lost run wrote its result: %s %q", statuses(all), all[2].Content)
	}
	var status string
	if err := e.db.QueryRow("SELECT status FROM agent_rpc_calls").Scan(&status); err != nil || status != string(agent.CallRunning) {
		t.Fatalf("call after lost lock: %q %v", status, err)
	}
}
