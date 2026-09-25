package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func serve(t *testing.T, contentType, body string) *llmClient {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return &llmClient{endpoint: srv.URL, http: srv.Client()}
}

func TestStreamToolCalls(t *testing.T) {
	body := `data: {"id":"r1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"think "}}]}

: keep-alive
data: {"choices":[{"index":0,"delta":{"content":"Hel"}}]}
data: {"choices":[{"index":0,"delta":{"content":"lo","tool_calls":[{"index":0,"id":"a","type":"function","function":{"name":"json_rpc","arguments":"{\"method\""}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"x\"}"}}]}}]}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"b","function":{"name":"json_rpc","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}
data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}
data: [DONE]
`
	c := serve(t, "text/event-stream", body)
	var got []string
	res, err := c.stream(context.Background(), chatRequest{}, nil, func(content, reasoning string) error {
		got = append(got, content+"|"+reasoning)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if res.ID != "r1" || res.Content != "Hello" || res.Reasoning != "think " || res.FinishReason != "tool_calls" || string(res.Usage) != `{"prompt_tokens":5,"completion_tokens":2}` {
		t.Fatalf("%+v", res)
	}
	if len(res.ToolCalls) != 2 || res.ToolCalls[0].Function.Arguments != `{"method":"x"}` || res.ToolCalls[1].ID != "b" {
		t.Fatalf("tool calls %+v", res.ToolCalls)
	}
	if strings.Join(got, ",") != "|think ,Hel|,lo|" {
		t.Fatalf("deltas %v", got)
	}
}

func TestStreamNonStreamingFallback(t *testing.T) {
	c := serve(t, "application/json", "{\n \"id\":\"x\",\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"},\"finish_reason\":\"stop\"}],\n \"usage\":{\"prompt_tokens\":1}\n}\n")
	res, err := c.stream(context.Background(), chatRequest{}, nil, nil)
	if err != nil || res.Content != "hi" || res.FinishReason != "stop" {
		t.Fatalf("%+v %v", res, err)
	}
}

func TestStreamBrokenAndErrors(t *testing.T) {
	// Cut off after output: not retried, reported as a broken stream.
	c := serve(t, "text/event-stream", "data: {\"choices\":[{\"delta\":{\"content\":\"par\"}}]}\n")
	c.maxRetries = 2
	_, err := c.stream(context.Background(), chatRequest{}, nil, nil)
	var broken *streamBrokenError
	if !errors.As(err, &broken) || !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("got %v", err)
	}
	// In-stream error object.
	c = serve(t, "text/event-stream", "data: {\"error\":{\"message\":\"quota\"}}\n")
	var apiErr *APIError
	if _, err := c.stream(context.Background(), chatRequest{}, nil, nil); !errors.As(err, &apiErr) || !strings.Contains(apiErr.Body, "quota") {
		t.Fatalf("got %v", err)
	}
}

func TestStreamIdleTimeout(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n")
		w.(http.Flusher).Flush()
		select {
		case <-r.Context().Done():
		case <-time.After(5 * time.Second):
		}
	}))
	defer srv.Close()
	c := &llmClient{endpoint: srv.URL, http: srv.Client(), idleTimeout: 200 * time.Millisecond}
	start := time.Now()
	_, err := c.stream(context.Background(), chatRequest{}, nil, nil)
	if !errors.Is(err, errIdleTimeout) || time.Since(start) > 3*time.Second {
		t.Fatalf("got %v after %v", err, time.Since(start))
	}
}
