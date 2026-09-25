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

func chatEndpoint(baseURL string) string {
	u := strings.TrimRight(baseURL, "/")
	if strings.HasSuffix(u, "/chat/completions") {
		return u
	}
	return u + "/chat/completions"
}
