package agent

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// APIError is returned when the LLM endpoint responds with an error.
type APIError struct {
	StatusCode int
	Body       string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("agent: llm api error (status %d): %s", e.StatusCode, e.Body)
}

type llmClient struct {
	endpoint    string
	apiKey      string
	headers     map[string]string
	http        *http.Client
	maxRetries  int
	idleTimeout time.Duration
}

type chatRequest struct {
	Model               string            `json:"model"`
	Messages            []json.RawMessage `json:"messages"`
	Tools               []json.RawMessage `json:"tools,omitempty"`
	MaxTokens           int               `json:"max_tokens,omitempty"`
	MaxCompletionTokens int               `json:"max_completion_tokens,omitempty"`
	Stream              bool              `json:"stream"`
	StreamOptions       *streamOptions    `json:"stream_options,omitempty"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

// streamResult is the assembled outcome of a streamed completion.
type streamResult struct {
	ID           string
	FinishReason string
	Content      string
	Reasoning    string
	ToolCalls    []ToolCall
	Usage        json.RawMessage
}

type deltaToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type streamChunk struct {
	ID      string `json:"id"`
	Choices []struct {
		Delta struct {
			Content          string          `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			Reasoning        string          `json:"reasoning"`
			ToolCalls        []deltaToolCall `json:"tool_calls"`
		} `json:"delta"`
		// Message is set when a provider ignores "stream" and answers in one piece.
		Message *struct {
			Content          json.RawMessage `json:"content"`
			ReasoningContent string          `json:"reasoning_content"`
			Reasoning        string          `json:"reasoning"`
			ToolCalls        []deltaToolCall `json:"tool_calls"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
	Error json.RawMessage `json:"error"`
}

// deltaFunc receives text as it streams in. Returning an error aborts the stream.
type deltaFunc func(content, reasoning string) error

// errStreamStarted marks errors after output was received; those are not retried.
type streamBrokenError struct{ err error }

func (e *streamBrokenError) Error() string { return "agent: llm stream interrupted: " + e.err.Error() }
func (e *streamBrokenError) Unwrap() error { return e.err }

var errIdleTimeout = errors.New("agent: llm stream idle timeout")

func (c *llmClient) stream(ctx context.Context, req chatRequest, extra map[string]any, onDelta deltaFunc) (*streamResult, error) {
	req.Stream = true
	req.StreamOptions = &streamOptions{IncludeUsage: true}
	body, err := buildBody(req, extra)
	if err != nil {
		return nil, err
	}
	var lastErr error
	for attempt := 0; attempt <= c.maxRetries; attempt++ {
		if attempt > 0 {
			t := time.NewTimer(time.Duration(1<<(attempt-1)) * time.Second)
			select {
			case <-ctx.Done():
				t.Stop()
				return nil, ctx.Err()
			case <-t.C:
			}
		}
		res, retry, err := c.doStream(ctx, body, onDelta)
		if err == nil {
			return res, nil
		}
		lastErr = err
		if !retry || ctx.Err() != nil {
			break
		}
	}
	return nil, lastErr
}

func (c *llmClient) doStream(parent context.Context, body []byte, onDelta deltaFunc) (*streamResult, bool, error) {
	ctx, cancel := context.WithCancelCause(parent)
	defer cancel(nil)
	var idle *time.Timer
	if c.idleTimeout > 0 {
		idle = time.AfterFunc(c.idleTimeout, func() { cancel(errIdleTimeout) })
		defer idle.Stop()
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "text/event-stream")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}
	httpResp, err := c.http.Do(httpReq)
	if err != nil {
		if cause := context.Cause(ctx); cause == errIdleTimeout {
			err = cause
		}
		return nil, true, fmt.Errorf("agent: llm request: %w", err)
	}
	defer httpResp.Body.Close()
	if httpResp.StatusCode/100 != 2 {
		data, _ := io.ReadAll(io.LimitReader(httpResp.Body, 64<<10))
		retry := httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500
		return nil, retry, &APIError{StatusCode: httpResp.StatusCode, Body: truncate(string(data), 2000)}
	}

	acc := &streamAccumulator{}
	started := false // after the first delta, failures are not retried
	fail := func(err error) (*streamResult, bool, error) {
		if cause := context.Cause(ctx); cause == errIdleTimeout {
			err = cause
		}
		if started {
			return nil, false, &streamBrokenError{err}
		}
		return nil, true, err
	}
	r := bufio.NewReader(httpResp.Body)
	done := false
	for !done {
		line, err := r.ReadString('\n')
		if idle != nil {
			idle.Reset(c.idleTimeout)
		}
		line = strings.TrimRight(line, "\r\n")
		payload := ""
		switch {
		case strings.HasPrefix(line, "data:"):
			payload = strings.TrimSpace(line[len("data:"):])
		case strings.HasPrefix(line, "{"):
			payload = line // plain JSON body (provider ignored "stream")
		}
		if payload == "[DONE]" {
			break
		}
		if payload != "" {
			var chunk streamChunk
			if jerr := json.Unmarshal([]byte(payload), &chunk); jerr != nil {
				if strings.HasPrefix(line, "{") && err == nil {
					// A multi-line JSON body: read the rest and parse it whole.
					rest, _ := io.ReadAll(r)
					payload += string(rest)
					if jerr = json.Unmarshal([]byte(payload), &chunk); jerr == nil {
						done = true
					}
				}
				if jerr != nil {
					return fail(fmt.Errorf("agent: decode llm stream chunk: %w", jerr))
				}
			}
			if len(chunk.Error) > 0 && string(chunk.Error) != "null" {
				// Proxies report upstream failures inside a 200 stream; honour
				// the embedded code so transient ones are retried.
				code := errorCode(chunk.Error)
				status := httpResp.StatusCode
				if code >= 400 {
					status = code
				}
				retry := !started && (code == http.StatusTooManyRequests || code >= 500)
				return nil, retry, &APIError{StatusCode: status, Body: truncate(string(chunk.Error), 2000)}
			}
			content, reasoning := acc.add(&chunk)
			if content != "" || reasoning != "" || len(chunk.Choices) > 0 {
				started = true
			}
			if (content != "" || reasoning != "") && onDelta != nil {
				if derr := onDelta(content, reasoning); derr != nil {
					return nil, false, derr
				}
			}
		}
		if err != nil {
			if err == io.EOF && acc.FinishReason != "" {
				break
			}
			if err == io.EOF {
				err = io.ErrUnexpectedEOF
			}
			return fail(err)
		}
	}
	return acc.result(), false, nil
}

type streamAccumulator struct {
	streamResult
	content, reasoning strings.Builder
}

func (a *streamAccumulator) add(ch *streamChunk) (content, reasoning string) {
	if ch.ID != "" {
		a.ID = ch.ID
	}
	if len(ch.Usage) > 0 && string(ch.Usage) != "null" {
		a.Usage = ch.Usage
	}
	for _, c := range ch.Choices {
		content = c.Delta.Content
		reasoning = c.Delta.ReasoningContent + c.Delta.Reasoning
		calls := c.Delta.ToolCalls
		if m := c.Message; m != nil {
			content = contentText(m.Content)
			reasoning = m.ReasoningContent + m.Reasoning
			calls = m.ToolCalls
			for i := range calls {
				calls[i].Index = i
			}
		}
		a.content.WriteString(content)
		a.reasoning.WriteString(reasoning)
		for _, d := range calls {
			a.addToolCall(d)
		}
		if c.FinishReason != "" {
			a.FinishReason = c.FinishReason
		}
		break // only the first choice is used
	}
	return content, reasoning
}

func (a *streamAccumulator) addToolCall(d deltaToolCall) {
	idx := d.Index
	// Some providers omit "index"; a new id at an occupied slot is a new call.
	if idx < len(a.ToolCalls) && d.ID != "" && a.ToolCalls[idx].ID != "" && a.ToolCalls[idx].ID != d.ID {
		idx = len(a.ToolCalls)
	}
	for len(a.ToolCalls) <= idx {
		a.ToolCalls = append(a.ToolCalls, ToolCall{Type: "function"})
	}
	tc := &a.ToolCalls[idx]
	if d.ID != "" {
		tc.ID = d.ID
	}
	if d.Type != "" {
		tc.Type = d.Type
	}
	if tc.Function.Name == "" {
		tc.Function.Name = d.Function.Name
	}
	tc.Function.Arguments += d.Function.Arguments
}

func (a *streamAccumulator) result() *streamResult {
	r := a.streamResult
	r.Content, r.Reasoning = a.content.String(), a.reasoning.String()
	return &r
}

func buildBody(req chatRequest, extra map[string]any) ([]byte, error) {
	if len(extra) == 0 {
		return marshalJSON(req)
	}
	base, err := marshalJSON(req)
	if err != nil {
		return nil, err
	}
	m := map[string]json.RawMessage{}
	if err := json.Unmarshal(base, &m); err != nil {
		return nil, err
	}
	for k, v := range extra {
		if k == "model" || k == "messages" || k == "tools" || k == "stream" {
			continue
		}
		if v == nil { // nil removes a default field, e.g. "stream_options"
			delete(m, k)
			continue
		}
		b, err := marshalJSON(v)
		if err != nil {
			return nil, fmt.Errorf("agent: extra body %q: %w", k, err)
		}
		m[k] = b
	}
	return marshalJSON(m)
}

// marshalJSON encodes without HTML escaping so stored raw JSON stays readable
// and identical on every replay.
func marshalJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func chatEndpoint(baseURL string) string {
	u := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(u, "/chat/completions") {
		return u
	}
	return u + "/chat/completions"
}

// errorCode extracts a numeric "code" from an error object, if any.
func errorCode(raw json.RawMessage) int {
	var e struct {
		Code json.RawMessage `json:"code"`
	}
	if json.Unmarshal(raw, &e) != nil {
		return 0
	}
	var n int
	if json.Unmarshal(e.Code, &n) == nil {
		return n
	}
	var s string
	if json.Unmarshal(e.Code, &s) == nil {
		n, _ = strconv.Atoi(s)
	}
	return n
}
