package agent

import (
	"encoding/json"
	"strconv"
	"strings"
)

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
