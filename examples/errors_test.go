package examples

import (
	"context"
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
}
