package examples

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
)

// requestShape returns the role/content of each message of a request, for
// asserting what the model was shown.
func requestShape(msgs []json.RawMessage) []string {
	var out []string
	for _, m := range msgs {
		var v struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		}
		_ = json.Unmarshal(m, &v)
		var s string
		_ = json.Unmarshal(v.Content, &s)
		if v.Role == "system" {
			s = "…"
		} else if len(s) > 40 {
			s = s[:40]
		}
		out = append(out, v.Role+":"+s)
	}
	return out
}

func compactionEnv(t *testing.T, mode agent.CompactionMode) (*env, *agent.Agent) {
	e := setup(t)
	e.cfg.ContextLength = 10000 // compaction at 8000 estimated tokens
	e.cfg.MaxOutputTokens = 500
	e.cfg.Compaction = mode
	return e, e.newAgent(t, "")
}

func TestCompactionAnchor(t *testing.T) {
	e, a := compactionEnv(t, "") // default = anchor
	ctx := context.Background()
	e.llm.pushReply(reply{msg: text("A"), hangAfter: -1, prompt: 500})
	e.llm.pushReply(reply{msg: text("B"), hangAfter: -1, prompt: 8500}) // conversation is now large
	e.llm.pushReply(reply{msg: text("C"), hangAfter: -1, prompt: 600})
	e.llm.pushReply(reply{msg: text("D"), hangAfter: -1, prompt: 700})
	for _, p := range []string{"a", "b", "c", "d"} {
		if _, err := a.Chat(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	// Before "c" the estimate (8600+) crossed 80%: the context restarts at "c".
	if got := strings.Join(requestShape(e.llm.request(2)), " | "); got != "system:… | user:c" {
		t.Fatalf("request after compaction: %s", got)
	}
	// Between compactions the prefix is stable, so the provider cache keeps hitting.
	c, d := e.llm.request(2), e.llm.request(3)
	for i := range c {
		if !bytes.Equal(c[i], d[i]) {
			t.Fatalf("prefix changed at %d", i)
		}
	}
	if got := strings.Join(requestShape(d), " | "); got != "system:… | user:c | assistant:C | user:d" {
		t.Fatal(got)
	}
	// The marker is stored for the UI, and nothing was deleted.
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	var marker *agent.Message
	for i := range all {
		if all[i].Kind == agent.MessageKindCompaction {
			marker = &all[i]
		}
	}
	if marker == nil || marker.Status != agent.MessageExcluded || marker.RefSeq != all[4].Seq || len(all) != 9 {
		t.Fatalf("marker %+v, %d messages", marker, len(all))
	}
	// A fresh agent on the same session resumes from the compaction point.
	e.llm.push(text("E"))
	if _, err := e.newAgent(t, a.SessionID()).Chat(ctx, "e"); err != nil {
		t.Fatal(err)
	}
	if got := requestShape(e.llm.request(4)); got[1] != "user:c" {
		t.Fatalf("resumed session does not start at the compaction point: %v", got)
	}
}

func TestCompactionSummary(t *testing.T) {
	e, a := compactionEnv(t, agent.CompactionSummary)
	ctx := context.Background()
	e.llm.pushReply(reply{msg: text("A"), hangAfter: -1, prompt: 500})
	e.llm.pushReply(reply{msg: text("B"), hangAfter: -1, prompt: 8500})
	e.llm.pushReply(reply{msg: text("user said a and b; assistant answered A and B"), hangAfter: -1, prompt: 300}) // the summary call
	e.llm.pushReply(reply{msg: text("C"), hangAfter: -1, prompt: 600})
	for _, p := range []string{"a", "b", "c"} {
		if _, err := a.Chat(ctx, p); err != nil {
			t.Fatal(err)
		}
	}
	// The summarizer saw the dropped messages as a transcript, with no tools.
	sum := e.llm.request(2)
	if len(sum) != 2 || !strings.Contains(string(sum[1]), "User: a") || !strings.Contains(string(sum[1]), "Assistant: B") {
		t.Fatalf("summary request: %s", sum)
	}
	if _, ok := e.llm.requests[2]["tools"]; ok {
		t.Fatal("summary request must not offer tools")
	}
	// The main request carries the summary, then the current turn.
	got := requestShape(e.llm.request(3))
	if len(got) != 3 || !strings.HasPrefix(got[1], "user:<conversation_summary>") || got[2] != "user:c" {
		t.Fatalf("request after summary: %v", got)
	}
	// The summary call is billed like any other.
	recs, _ := e.client.ListUsage(ctx, a.SessionID())
	if len(recs) != 4 {
		t.Fatalf("want 4 usage records (incl. summary), got %d", len(recs))
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	for _, m := range all {
		if m.Kind == agent.MessageKindCompaction && (m.Status != agent.MessageDone || !strings.Contains(m.Content, "assistant answered")) {
			t.Fatalf("summary message %+v", m)
		}
	}
}

func TestCompactionNoProgressAndOff(t *testing.T) {
	// A single huge turn cannot be compacted (the window already starts at
	// its user message); the run continues until the budget runs out.
	e, a := compactionEnv(t, "")
	ctx := context.Background()
	e.llm.pushReply(reply{msg: toolCall("t1", "get_order", `{"order_id":"O"}`), hangAfter: -1, prompt: 9990})
	_, err := a.Chat(ctx, "big")
	if err == nil || !strings.Contains(err.Error(), "context length") {
		t.Fatalf("want ErrContextLengthExceeded, got %v", err)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	for _, m := range all {
		if m.Kind == agent.MessageKindCompaction {
			t.Fatal("nothing to drop, no compaction expected")
		}
	}

	e2, a2 := compactionEnv(t, agent.CompactionOff)
	e2.llm.pushReply(reply{msg: text("A"), hangAfter: -1, prompt: 8500})
	e2.llm.push(text("B"))
	_, _ = a2.Chat(ctx, "a")
	if _, err := a2.Chat(ctx, "b"); err != nil {
		t.Fatal(err)
	}
	if n := len(e2.llm.request(1)); n != 4 { // system, a, A, b: nothing dropped
		t.Fatalf("off mode dropped messages: %d", n)
	}
}

// recordingStore records the start seq of every "messages from seq" query.
type recordingStore struct {
	agent.Store
	mu    sync.Mutex
	froms []int64
}

func (s *recordingStore) Query(ctx context.Context, q string, args ...any) (agent.Rows, error) {
	if strings.Contains(q, "seq >= ?") {
		s.mu.Lock()
		s.froms = append(s.froms, args[1].(int64))
		s.mu.Unlock()
	}
	return s.Store.Query(ctx, q, args...)
}

// After a compaction, a step reads only from the compaction point on.
func TestWindowLoadsFromCompactionPoint(t *testing.T) {
	e := setup(t)
	rs := &recordingStore{Store: e.store}
	e.cfg.Store = rs
	e.cfg.ContextLength = 10000
	client, err := agent.NewClient(context.Background(), e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	a, _ := client.Agent(context.Background(), "", agent.AgentOptions{})
	e.llm.pushReply(reply{msg: text("A"), hangAfter: -1, prompt: 8500})
	e.llm.push(text("B"))
	for _, p := range []string{"a", "b"} {
		if _, err := a.Chat(context.Background(), p); err != nil {
			t.Fatal(err)
		}
	}
	all, _ := client.LatestMessages(context.Background(), a.SessionID(), 100)
	var refSeq int64
	for _, m := range all {
		if m.Kind == agent.MessageKindCompaction {
			refSeq = m.RefSeq
		}
	}
	rs.mu.Lock()
	last := rs.froms[len(rs.froms)-1]
	rs.mu.Unlock()
	if refSeq == 0 || last != refSeq {
		t.Fatalf("last window query started at seq %d, compaction point is %d", last, refSeq)
	}
}
