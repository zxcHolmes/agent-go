package examples

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

func TestChatToolLoop(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	sid, err := e.client.CreateSession(ctx, map[string]any{"owner": "u1"})
	if err != nil {
		t.Fatal(err)
	}
	a := e.newAgent(t, sid)
	e.llm.push(
		toolCall("c1", "get_order", `{}`),                 // invalid params -> error back to model
		toolCall("c2", "get_order", `{"order_id":"O-1"}`), // ok
		text("Order O-1 found."),
	)
	res, err := a.Chat(ctx, "where is my order?")
	if err != nil {
		t.Fatal(err)
	}
	if res.StopReason != agent.StopCompleted || res.Status != agent.StatusIdle || res.Reply() != "Order O-1 found." {
		t.Fatalf("result %+v", res)
	}
	if got := roles(res.Messages); got != "user,assistant,tool,assistant,tool,assistant" {
		t.Fatal(got)
	}
	if got := statuses(res.Messages); got != "done,done,done,done,done,done" {
		t.Fatal(got)
	}
	if c := res.Messages[2].Content; !strings.Contains(c, `"code":-32602`) || !strings.Contains(c, "order_id is required") {
		t.Fatalf("invalid params not reported: %s", c)
	}
	if c := res.Messages[4].Content; c != `{"jsonrpc":"2.0","id":1,"result":{"order_id":"O-1","user":"u1"}}` {
		t.Fatalf("tool result: %s", c)
	}

	// Prefix cache: each request's messages must be a byte-exact prefix of the next.
	var prev []json.RawMessage
	for i := range e.llm.requests {
		msgs := e.llm.request(i)
		for j := range prev {
			if !bytes.Equal(prev[j], msgs[j]) {
				t.Fatalf("request %d message %d changed:\n%s\n%s", i, j, prev[j], msgs[j])
			}
		}
		prev = msgs
	}
	if !bytes.Contains(e.llm.requests[0]["stream"], []byte("true")) || !bytes.Contains(e.llm.requests[0]["tools"], []byte(`"name":"json_rpc"`)) {
		t.Fatal("request must stream and carry the tool")
	}

	// Usage & billing: 3 calls x (200 uncached in, 800 cached, 100 out).
	sum, err := e.client.Usage(ctx, a.SessionID())
	if err != nil {
		t.Fatal(err)
	}
	wantCredits := 3 * (200*100.0/1e6 + 800*10.0/1e6 + 100*1000.0/1e6)
	if sum.Calls != 3 || sum.Usage.CachedTokens != 2400 || sum.Usage.ReasoningTokens != 60 || abs(sum.Cost.Total-wantCredits) > 1e-9 {
		t.Fatalf("usage %+v", sum)
	}

	// Cursor pagination.
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	latest, _ := e.client.LatestMessages(ctx, a.SessionID(), 2)
	if len(all) != 6 || roles(latest) != "tool,assistant" || latest[1].ID != all[5].ID {
		t.Fatalf("latest %s", roles(latest))
	}
	before, _ := e.client.MessagesBefore(ctx, a.SessionID(), all[3].ID, 2)
	after, _ := e.client.MessagesAfter(ctx, a.SessionID(), all[3].ID, 10)
	if len(before) != 2 || before[0].ID != all[1].ID || len(after) != 2 || after[0].ID != all[4].ID {
		t.Fatal("before/after")
	}

	// Resume the session with a fresh agent.
	e.llm.push(text("again"))
	res, err = e.newAgent(t, sid).Chat(ctx, "thanks")
	if err != nil || res.Reply() != "again" {
		t.Fatal(res, err)
	}
	if n := len(e.llm.request(3)); n != 8 { // system + 6 history + new user
		t.Fatalf("resumed request has %d messages", n)
	}
}

func TestStreamingFlush(t *testing.T) {
	e := setup(t)
	e.cfg.StreamFlushInterval = 150 * time.Millisecond
	var writes, deltas atomic.Int32
	e.cfg.OnMessage = func(ctx context.Context, m agent.Message) {
		if m.Role == "assistant" {
			writes.Add(1)
		}
	}
	e.cfg.OnStream = func(ctx context.Context, id, content, reasoning string) { deltas.Add(1) }
	ctx := context.Background()
	a := e.newAgent(t, "")
	answer := strings.Repeat("流式输出", 6) // 24 runes -> 8 chunks
	e.llm.pushReply(reply{msg: text(answer), delay: 60 * time.Millisecond, hangAfter: -1})

	done := make(chan error)
	go func() {
		_, err := a.Chat(ctx, "hi")
		done <- err
	}()

	// Poll like a frontend would: cursor = the user message, fetch everything after it.
	var user agent.Message
	waitFor(t, "user message", func() bool {
		ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 5)
		for _, m := range ms {
			if m.Role == "user" {
				user = m
				return true
			}
		}
		return false
	})
	var seen []string
	waitFor(t, "stream to finish", func() bool {
		ms, _ := e.client.MessagesAfter(ctx, a.SessionID(), user.ID, 10)
		if len(ms) == 0 {
			return false
		}
		m := ms[0]
		if len(seen) == 0 || seen[len(seen)-1] != m.Content {
			seen = append(seen, m.Content)
		}
		if m.Status == agent.MessageStreaming && !strings.HasPrefix(answer, m.Content) {
			t.Fatalf("partial content %q is not a prefix", m.Content)
		}
		return m.Status == agent.MessageDone
	})
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	if seen[len(seen)-1] != answer || len(seen) < 3 {
		t.Fatalf("expected several growing snapshots, got %q", seen)
	}
	// 8 deltas over ~480ms with a 150ms interval: far fewer writes than deltas.
	if deltas.Load() != 8 || writes.Load() < 3 || writes.Load() > 6 {
		t.Fatalf("deltas=%d writes=%d", deltas.Load(), writes.Load())
	}
}

func TestClientOverridesAndReadOnly(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	client, err := agent.NewClient(ctx, e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, err := client.Agent(ctx, "", agent.AgentOptions{Override: func(c *agent.Config) {
		c.SystemPrompt = "you are a pirate"
		c.Methods = nil
	}})
	if err != nil {
		t.Fatal(err)
	}
	e.llm.push(text("arr"))
	if _, err := a.Chat(ctx, "hi"); err != nil {
		t.Fatal(err)
	}
	msgs := e.llm.request(0)
	if !strings.Contains(string(msgs[0]), "you are a pirate") || strings.Contains(string(msgs[0]), "get_order") {
		t.Fatalf("override not applied: %s", msgs[0])
	}
	if _, ok := e.llm.requests[0]["tools"]; ok {
		t.Fatal("no methods -> no tools")
	}

	// A client without LLM settings still serves reads and billing.
	ro, err := agent.NewClient(ctx, agent.Config{Store: e.store})
	if err != nil {
		t.Fatal(err)
	}
	if ms, err := ro.LatestMessages(ctx, a.SessionID(), 5); err != nil || len(ms) != 2 {
		t.Fatal(ms, err)
	}
	if sum, _, err := ro.UnbilledUsage(ctx, a.SessionID()); err != nil || sum.Calls != 1 {
		t.Fatal(sum, err)
	}
	if _, err := ro.Agent(ctx, "", agent.AgentOptions{}); err == nil {
		t.Fatal("agent without BaseURL/Model must fail")
	}
}
