package agent

import (
	"bytes"
	"context"
	"encoding/json"
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
	endpoint   string
	apiKey     string
	headers    map[string]string
	http       *http.Client
	maxRetries int
}

type chatRequest struct {
	Model               string            `json:"model"`
	Messages            []json.RawMessage `json:"messages"`
	Tools               []json.RawMessage `json:"tools,omitempty"`
	MaxTokens           int               `json:"max_tokens,omitempty"`
	MaxCompletionTokens int               `json:"max_completion_tokens,omitempty"`
}

type chatResponse struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message      json.RawMessage `json:"message"`
		FinishReason string          `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
	Error json.RawMessage `json:"error"`
}

func (c *llmClient) complete(ctx context.Context, req chatRequest, extra map[string]any) (*chatResponse, error) {
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
		resp, retry, err := c.do(ctx, body)
		if err == nil {
			return resp, nil
		}
		lastErr = err
		if !retry || ctx.Err() != nil {
			break
		}
	}
	return nil, lastErr
}

func (c *llmClient) do(ctx context.Context, body []byte) (*chatResponse, bool, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, c.endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if c.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+c.apiKey)
	}
	for k, v := range c.headers {
		httpReq.Header.Set(k, v)
	}
	httpResp, err := c.http.Do(httpReq)
	if err != nil {
		return nil, true, fmt.Errorf("agent: llm request: %w", err)
	}
	defer httpResp.Body.Close()
	data, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, true, fmt.Errorf("agent: read llm response: %w", err)
	}
	if httpResp.StatusCode/100 != 2 {
		retry := httpResp.StatusCode == http.StatusTooManyRequests || httpResp.StatusCode >= 500
		return nil, retry, &APIError{StatusCode: httpResp.StatusCode, Body: truncate(string(data), 2000)}
	}
	var out chatResponse
	if err := json.Unmarshal(data, &out); err != nil {
		return nil, false, fmt.Errorf("agent: decode llm response: %w", err)
	}
	if len(out.Error) > 0 && string(out.Error) != "null" {
		return nil, false, &APIError{StatusCode: httpResp.StatusCode, Body: truncate(string(out.Error), 2000)}
	}
	if len(out.Choices) == 0 || len(out.Choices[0].Message) == 0 {
		return nil, false, &APIError{StatusCode: httpResp.StatusCode, Body: "no choices in response: " + truncate(string(data), 2000)}
	}
	return &out, false, nil
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
		if k == "model" || k == "messages" || k == "tools" {
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
