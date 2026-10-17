package examples

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

func TestLLMErrorAndContextLength(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	_, err := a.Chat(ctx, "hi") // no scripted reply -> 400
	var apiErr *agent.APIError
	if !errors.As(err, &apiErr) || apiErr.StatusCode != 400 {
		t.Fatalf("want APIError, got %v", err)
	}
	s, _ := a.Session(ctx)
	if s.Status != agent.StatusIdle || s.LastError == "" {
		t.Fatalf("%+v", s)
	}
	e.llm.push(text("hello"))
	if res, err := a.Continue(ctx); err != nil || res.Reply() != "hello" {
		t.Fatal(res, err)
	}

	e.cfg.ContextLength = 50
	small := e.newAgent(t, a.SessionID())
	if _, err := small.Chat(ctx, strings.Repeat("x", 500)); !errors.Is(err, agent.ErrContextLengthExceeded) {
		t.Fatalf("want ErrContextLengthExceeded, got %v", err)
	}
}

// Seen live: a proxy reported an upstream 502 inside an HTTP 200 stream.
func TestInStreamTransientErrorIsRetried(t *testing.T) {
	e := setup(t)
	e.cfg.MaxRetries = 2
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{streamErr: `{"code":502,"message":"Flex processing is temporarily unavailable"}`})
	e.llm.push(text("ok"))
	res, err := a.Chat(context.Background(), "hi")
	if err != nil || res.Reply() != "ok" || len(e.llm.requests) != 2 {
		t.Fatalf("%v %v requests=%d", res, err, len(e.llm.requests))
	}
	// A non-transient in-stream error is not retried.
	e.llm.pushReply(reply{streamErr: `{"code":400,"message":"bad request"}`})
	var ae *agent.APIError
	if _, err := a.Chat(context.Background(), "again"); !errors.As(err, &ae) || ae.StatusCode != 400 {
		t.Fatalf("got %v", err)
	}
	// An error the provider reports itself is not billed as a broken-off request.
	recs, _ := e.client.ListUsage(context.Background(), a.SessionID())
	for _, r := range recs {
		if r.FinishReason == agent.FinishInterrupted {
			t.Fatalf("a provider error must not be estimated: %+v", r)
		}
	}
}

// Seen live: a proxy reset the connection while the model was emitting a tool
// call. Nothing had been shown — tool-call fragments are never streamed out —
// so the request is simply retried, and the user never sees the break.
func TestToolCallStreamBreakIsRetried(t *testing.T) {
	e := setup(t)
	e.cfg.MaxRetries = 2
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: toolCall("c1", "get_order", `{"order_id":"A-1"}`), hangAfter: -1, cut: 1})
	e.llm.push(toolCall("c1", "get_order", `{"order_id":"A-1"}`), text("done"))
	res, err := a.Chat(context.Background(), "hi")
	if err != nil || res.Reply() != "done" || len(e.llm.requests) != 3 {
		t.Fatalf("%v %v requests=%d", res, err, len(e.llm.requests))
	}
	for _, m := range res.Messages {
		if m.Status == agent.MessageInterrupted {
			t.Fatalf("a retried stream left an interrupted message: %+v", m)
		}
	}
}

// A stream that breaks after text was shown is not retried inside the request
// (the text is already stored), but the run retries the step once: the partial
// reply is kept and replayed, and the model continues from it.
func TestStreamBreakAfterTextRetriesTheStepOnce(t *testing.T) {
	e := setup(t)
	a := e.newAgent(t, "")
	e.llm.pushReply(reply{msg: text("Let me look that up for you"), hangAfter: -1, cut: 2})
	e.llm.push(text("ok"))
	res, err := a.Chat(context.Background(), "hi")
	if err != nil || res.Reply() != "ok" || len(e.llm.requests) != 2 {
		t.Fatalf("%v %v requests=%d", res, err, len(e.llm.requests))
	}
	if got := statuses(res.Messages); !strings.Contains(got, string(agent.MessageInterrupted)) {
		t.Fatalf("the partial reply should be kept as interrupted: %s", got)
	}
	if last := e.llm.request(1); !strings.Contains(string(last[len(last)-1]), "Let me") {
		t.Fatalf("the retry should replay the partial reply, got %s", last[len(last)-1])
	}

	// Only once: a second break in the same step ends the run.
	e.llm.pushReply(reply{msg: text("Let me look that up for you"), hangAfter: -1, cut: 2})
	e.llm.pushReply(reply{msg: text("Let me look that up for you"), hangAfter: -1, cut: 2})
	if _, err := a.Chat(context.Background(), "again"); err == nil || !strings.Contains(err.Error(), "llm stream interrupted") {
		t.Fatalf("want a broken stream error, got %v", err)
	}
}

// A reply cut off at max_tokens while writing a tool call: the call does not
// run, and the model is told it hit the output limit and to split the work,
// not handed a parse error it would answer by resending the same call.
func TestToolCallCutOffAtOutputLimit(t *testing.T) {
	e := setup(t)
	ctx := context.Background()
	a := e.newAgent(t, "")
	cut, _ := json.Marshal(`{"jsonrpc":"2.0","method":"get_order","params":{"order_id":"O`)
	e.llm.pushReply(reply{msg: `{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"json_rpc","arguments":` + string(cut) + `}}]}`,
		hangAfter: -1, finish: "length"})
	e.llm.push(text("ok, in parts"))
	if _, err := a.Chat(ctx, "write it"); err != nil {
		t.Fatal(err)
	}
	all, _ := e.client.LatestMessages(ctx, a.SessionID(), 100)
	var result string
	for _, m := range all {
		if m.Role == "tool" && m.ToolCallID == "t1" {
			result = m.Content
		}
	}
	if !strings.Contains(result, "output limit") || !strings.Contains(result, "Split the work") {
		t.Fatalf("tool result: %s", result)
	}
}
