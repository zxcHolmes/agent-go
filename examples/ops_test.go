package examples

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"strings"
	"sync"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
)

var errNoCredits = errors.New("insufficient credits")

func TestBeforeLLMCallBudgetGuard(t *testing.T) {
	e := setup(t)
	var infos []agent.LLMCallInfo
	e.cfg.BeforeLLMCall = func(ctx context.Context, info agent.LLMCallInfo) error {
		infos = append(infos, info)
		if len(infos) == 2 {
			return errNoCredits
		}
		return nil
	}
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "get_order", `{"order_id":"O"}`), text("never sent"))
	_, err := a.Chat(ctx, "go")
	if !errors.Is(err, errNoCredits) {
		t.Fatalf("want the hook's error, got %v", err)
	}
	if len(e.llm.requests) != 1 {
		t.Fatalf("the rejected request was sent anyway (%d requests)", len(e.llm.requests))
	}
	i := infos[0]
	if i.Purpose != "chat" || i.Model != "fake" || i.EstimatedPromptTokens <= 0 || i.MaxTokens <= 0 || i.MaxCost.Total <= 0 {
		t.Fatalf("info %+v", i)
	}
	s, _ := a.Session(ctx)
	if s.Status != agent.StatusIdle || !strings.Contains(s.LastError, "insufficient credits") {
		t.Fatalf("%+v", s)
	}
	e.checkValid(t, a.SessionID())
}

func TestBeforeLLMCallCoversSummaries(t *testing.T) {
	e := setup(t)
	e.cfg.ContextLength, e.cfg.Compaction = 10000, agent.CompactionSummary
	e.cfg.BeforeLLMCall = func(ctx context.Context, info agent.LLMCallInfo) error {
		if info.Purpose == "summary" {
			return errNoCredits
		}
		return nil
	}
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("A"), hangAfter: -1, prompt: 8500})
	_, _ = a.Chat(ctx, "a")
	if _, err := a.Chat(ctx, "b"); !errors.Is(err, errNoCredits) {
		t.Fatalf("summary call must be guarded too, got %v", err)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 20)
	for _, m := range all {
		if m.Kind == agent.MessageKindCompaction {
			t.Fatal("a rejected summary must not compact")
		}
	}
}

func TestRPCTimeout(t *testing.T) {
	e := setup(t, agent.Method{
		Name:    "hangs",
		Timeout: 100 * time.Millisecond,
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			time.Sleep(2 * time.Second) // ignores ctx
			return "late", nil
		},
	}, agent.Method{
		Name: "slowish", // no timeout by default
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			time.Sleep(300 * time.Millisecond)
			return "fine", nil
		},
	})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "hangs", `{}`), toolCall("t2", "slowish", `{}`), text("done"))
	start := time.Now()
	if _, err := a.Chat(ctx, "go"); err != nil {
		t.Fatal(err)
	}
	if time.Since(start) > 1500*time.Millisecond {
		t.Fatalf("timeout not enforced: %v", time.Since(start))
	}
	var results []string
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 20)
	for _, m := range all {
		if m.Role == "tool" {
			results = append(results, m.Content)
		}
	}
	if !strings.Contains(results[0], `"code":-32004`) || !strings.Contains(results[0], "timed out after 100ms") || !strings.Contains(results[1], `"result":"fine"`) {
		t.Fatalf("%v", results)
	}
}

type logBuf struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (l *logBuf) Write(p []byte) (int, error) { l.mu.Lock(); defer l.mu.Unlock(); return l.b.Write(p) }
func (l *logBuf) String() string              { l.mu.Lock(); defer l.mu.Unlock(); return l.b.String() }

func TestLoggingLevelsAndPanic(t *testing.T) {
	for _, level := range []slog.Level{slog.LevelInfo, slog.LevelDebug} {
		e := setup(t, agent.Method{Name: "boom", Handler: func(ctx context.Context, c *agent.Call) (any, error) { panic("kaboom") }})
		var buf logBuf
		// The shared logger allows everything; the agent's LogLevel decides.
		e.cfg.Logger = slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelDebug}))
		e.cfg.LogLevel = level
		a := e.newAgent(t, "")
		e.llm.push(toolCall("t1", "boom", `{}`), text("sorry"))
		if _, err := a.Chat(context.Background(), "go"); err != nil {
			t.Fatal(err)
		}
		out := buf.String()
		for _, want := range []string{"run started", "run finished", "rpc handler panicked", "kaboom", "goroutine"} {
			if !strings.Contains(out, want) {
				t.Fatalf("level %s: log missing %q:\n%s", level, want, out)
			}
		}
		if hasDebug := strings.Contains(out, "llm request") && strings.Contains(out, "rpc call"); hasDebug != (level == slog.LevelDebug) {
			t.Fatalf("level %s: debug lines present=%v:\n%s", level, hasDebug, out)
		}
		// The model gets the panic value, never the stack.
		all, _ := e.client.LatestMessages(context.Background(), a.SessionID(), 10)
		if c := all[2].Content; !strings.Contains(c, "kaboom") || strings.Contains(c, "goroutine") {
			t.Fatalf("tool result: %s", c)
		}
	}
}

func TestCacheControl(t *testing.T) {
	e := setup(t)
	e.cfg.CacheControl = true
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(text("one"), text("two"))
	_, _ = a.Chat(ctx, "first")
	_, _ = a.Chat(ctx, "second")
	msgs := e.llm.request(1)
	marked := func(i int) bool { return strings.Contains(string(msgs[i]), `"cache_control":{"type":"ephemeral"}`) }
	if !marked(0) || !marked(len(msgs)-1) {
		t.Fatalf("system and last message must carry breakpoints:\n%s", msgs)
	}
	for i := 1; i < len(msgs)-1; i++ {
		if marked(i) {
			t.Fatalf("message %d should not be marked", i)
		}
	}
	if !strings.Contains(string(msgs[len(msgs)-1]), `"content":[{"cache_control":{"type":"ephemeral"},"text":"second","type":"text"}]`) {
		t.Fatalf("last: %s", msgs[len(msgs)-1])
	}
	// Stored messages are untouched.
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	for _, m := range ms {
		if strings.Contains(string(m.Raw), "cache_control") {
			t.Fatalf("stored raw modified: %s", m.Raw)
		}
	}
}

func TestStopPollInterval(t *testing.T) {
	started := make(chan struct{})
	e := setup(t, agent.Method{Name: "work", Handler: func(ctx context.Context, c *agent.Call) (any, error) {
		close(started)
		time.Sleep(600 * time.Millisecond) // ignores ctx
		return "worked", nil
	}})
	e.cfg.StopPollInterval = -1 // no polling: remote stops are seen between steps only
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "work", `{}`), text("unreachable"))
	done := make(chan *agent.RunResult)
	go func() {
		res, _ := a.Chat(ctx, "go")
		done <- res
	}()
	<-started
	_, _ = e.store.Exec(ctx, "UPDATE agent_sessions SET status = 'stopping' WHERE id = ?", a.SessionID())
	res := <-done
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if res.StopReason != agent.StopStopped || !strings.Contains(all[2].Content, `"result":"worked"`) {
		t.Fatalf("without polling the call finishes, then the stop applies: %s %s", res.StopReason, all[2].Content)
	}
}

// checkValid asserts every message is final and the session is idle.
func (e *env) checkValid(t *testing.T, sid string) {
	t.Helper()
	ms, _ := e.client.LatestMessages(context.Background(), sid, 1000)
	for _, m := range ms {
		if !m.Status.Final() {
			t.Fatalf("message %d not final", m.Seq)
		}
	}
}

// A handler that honours ctx returns ctx.Err() on timeout; the model must
// still see the timeout, not an internal "context deadline exceeded" error.
func TestRPCTimeoutWithCtxAwareHandler(t *testing.T) {
	e := setup(t, agent.Method{Name: "polite", Timeout: 50 * time.Millisecond, Handler: func(ctx context.Context, c *agent.Call) (any, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}})
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "polite", `{}`), text("ok"))
	for i := 0; i < 20; i++ { // the race is timing dependent
		if _, err := a.Chat(context.Background(), "go"); err != nil {
			t.Fatal(err)
		}
		ms, _ := e.client.LatestMessages(context.Background(), a.SessionID(), 2)
		if !strings.Contains(ms[0].Content, `"code":-32004`) {
			t.Fatalf("iteration %d: %s", i, ms[0].Content)
		}
		e.llm.push(toolCall("t1", "polite", `{}`), text("ok"))
	}
}
