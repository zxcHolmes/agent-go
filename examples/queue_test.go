package examples

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	agent "github.com/zxcHolmes/agent-go"
)

func TestQueueDuringRun(t *testing.T) {
	started := make(chan struct{})
	release := make(chan struct{})
	e := setup(t, agent.Method{
		Name: "slow",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			close(started)
			<-release
			return "done", nil
		},
	})
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "slow", `{}`), text("Handled both details."))
	done := make(chan *agent.RunResult)
	go func() {
		res, err := a.Chat(ctx, "do the task")
		if err != nil {
			t.Error(err)
		}
		done <- res
	}()
	<-started
	// The user adds details while the tool runs.
	id1, err := a.Enqueue(ctx, "detail one")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := e.client.Enqueue(ctx, a.SessionID(), "detail two"); err != nil {
		t.Fatal(err)
	}
	if q, _ := e.client.QueuedMessages(ctx, a.SessionID()); len(q) != 2 || q[0].ID != id1 || q[0].Content != "detail one" {
		t.Fatalf("queue: %+v", q)
	}
	close(release)
	<-done
	// Both arrive as separate user messages right after the tool result, in one request.
	got := requestShape(e.llm.request(1))
	if strings.Join(got[len(got)-3:], " | ") != "tool:{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\"done\"} | user:detail one | user:detail two" {
		t.Fatalf("request: %v", got)
	}
	if q, _ := e.client.QueuedMessages(ctx, a.SessionID()); len(q) != 0 {
		t.Fatal("queue not emptied")
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 20)
	if all[3].ID != "msg_"+id1 || all[3].Role != "user" {
		t.Fatalf("queued message stored as %+v", all[3])
	}
}

// A message queued while the model writes its final answer still gets an answer.
func TestQueueAfterFinalAnswerContinuesLoop(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("first answer here"), hangAfter: -1, delay: 50 * time.Millisecond})
	e.llm.push(text("answer to the follow-up"))
	var enqueued bool
	e.cfg.OnStream = func(ctx context.Context, id, c, r string) {
		if !enqueued {
			enqueued = true
			_, _ = e.client.Enqueue(ctx, a.SessionID(), "and one more thing")
		}
	}
	a = e.newAgent(t, a.SessionID())
	res, err := a.Chat(ctx, "question")
	if err != nil {
		t.Fatal(err)
	}
	if res.Reply() != "answer to the follow-up" || len(e.llm.requests) != 2 {
		t.Fatalf("reply %q after %d requests", res.Reply(), len(e.llm.requests))
	}
}

func TestQueueWhileIdleAndCancel(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	idKeep, _ := a.Enqueue(ctx, "queued earlier")
	idDrop, _ := a.Enqueue(ctx, "never mind this")
	if err := e.client.CancelQueued(ctx, a.SessionID(), idDrop); err != nil {
		t.Fatal(err)
	}
	e.llm.push(text("ok"))
	if _, err := a.Chat(ctx, "now"); err != nil {
		t.Fatal(err)
	}
	// Queued before the new prompt, cancelled one gone.
	if got := strings.Join(requestShape(e.llm.request(0))[1:], " | "); got != "user:queued earlier | user:now" {
		t.Fatalf("%s (kept %s)", got, idKeep)
	}
}

// A message the run already took can no longer be withdrawn, and the caller
// is told so instead of a silent success.
func TestCancelQueuedAfterSent(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	id, _ := a.Enqueue(ctx, "sent before cancel")
	e.llm.push(text("ok"))
	if _, err := a.Chat(ctx, "now"); err != nil {
		t.Fatal(err)
	}
	if err := a.CancelQueued(ctx, id); !errors.Is(err, agent.ErrQueuedMessageSent) {
		t.Fatalf("want ErrQueuedMessageSent, got %v", err)
	}
	// Unknown / already withdrawn ids are a no-op.
	if err := a.CancelQueued(ctx, "q_unknown"); err != nil {
		t.Fatal(err)
	}
}

// A row claimed by a run (taken, message not written yet) cannot be withdrawn;
// if that run crashed, the next run still sends it exactly once.
func TestCancelQueuedWhileTaken(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	id, _ := a.Enqueue(ctx, "claimed by a run")
	if _, err := e.store.Exec(ctx, "UPDATE agent_queued_messages SET status = 'taken' WHERE id = ?", id); err != nil {
		t.Fatal(err)
	}
	if err := a.CancelQueued(ctx, id); !errors.Is(err, agent.ErrQueuedMessageSent) {
		t.Fatalf("want ErrQueuedMessageSent, got %v", err)
	}
	if qs, _ := e.client.QueuedMessages(ctx, a.SessionID()); len(qs) != 0 {
		t.Fatalf("taken message still listed as withdrawable: %+v", qs)
	}
	e.llm.push(text("ok"))
	if _, err := a.Chat(ctx, "now"); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(requestShape(e.llm.request(0))[1:], " | "); got != "user:claimed by a run | user:now" {
		t.Fatal(got)
	}
}

func TestReminder(t *testing.T) {
	e := setup(t)
	e.cfg.Reminder = "Answer in French."
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(text("oui"), text("oui"))
	if _, err := a.Chat(ctx, "hello"); err != nil {
		t.Fatal(err)
	}
	// The model sees the reminder; the stored content is what the user typed.
	sent := string(e.llm.request(0)[1])
	if sent != `{"role":"user","content":"hello\n\n<reminder>\nAnswer in French.\n</reminder>"}` {
		t.Fatalf("sent: %s", sent)
	}
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 2)
	if ms[0].Content != "hello" {
		t.Fatalf("stored content %q", ms[0].Content)
	}
	// Multimodal content gets an extra text part.
	raw := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://x/y.png"}}]}`)
	if _, err := a.ChatMessage(ctx, raw); err != nil {
		t.Fatal(err)
	}
	sent = string(e.llm.request(1)[3])
	if !strings.HasSuffix(sent, `{"type":"text","text":"<reminder>\nAnswer in French.\n</reminder>"}]}`) {
		t.Fatalf("sent: %s", sent)
	}
}

func TestRPCResultLimit(t *testing.T) {
	e := setup(t, agent.Method{
		Name: "big",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			return strings.Repeat("数据", 50), nil // 100 characters, 300 bytes
		},
	}, agent.Method{
		Name: "bigerr",
		Handler: func(ctx context.Context, c *agent.Call) (any, error) {
			return nil, agent.InvalidParams("%s", strings.Repeat("e", 100))
		},
	})
	e.cfg.MaxRPCResultChars = 20
	ctx := context.Background()
	a := e.newAgent(t, "")
	e.llm.push(toolCall("t1", "big", `{}`), toolCall("t2", "bigerr", `{}`), text("ok"))
	if _, err := a.Chat(ctx, "go"); err != nil {
		t.Fatal(err)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 20)
	var results []string
	for _, m := range all {
		if m.Role == "tool" {
			results = append(results, m.Content)
			if !json.Valid([]byte(m.Content)) {
				t.Fatalf("truncated result is not valid JSON: %s", m.Content)
			}
		}
	}
	// 20 characters of the JSON-encoded string: the opening quote plus 19 CJK characters.
	if !strings.Contains(results[0], `"result":"\"数据数据数据数据数据数据数据数据数据数…[truncated: 82 more characters not shown]"`) {
		t.Fatalf("result: %s", results[0])
	}
	if !strings.Contains(results[1], `"message":"eeeeeeeeeeeeeeeeeeee…[truncated: 80 more characters not shown]"`) {
		t.Fatalf("error: %s", results[1])
	}
}
