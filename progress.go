package agent

import (
	"errors"
	"fmt"
	"strings"
)

// A model can fall into repeating itself: the same call again and again with
// the same result, or documents it already has read over and over. Seen live:
// a model wrote "I will read program.js with read_file now" and called
// read_doc for the same document, nine times in a row, until the step limit.
// A note in the result does not stop it — it already knows — so the run ends.
const (
	// sameStepsLimit is how many identical steps in a row (same calls, the
	// earlier ones with the same results) end the run, this one included.
	sameStepsLimit = 4
	// rereadLimit is how many read_doc answers in one turn may point at a copy
	// already in the context before the run ends.
	rereadLimit = 6
)

// errNoProgress makes the loop end the run with StopNoProgress.
var errNoProgress = errors.New("agent: no progress")

// stepSignature is a step's tool calls, by name and arguments.
func stepSignature(calls []ToolCall) string {
	var b strings.Builder
	for _, tc := range calls {
		b.WriteString(tc.Function.Name)
		b.WriteByte(0)
		b.WriteString(strings.TrimSpace(tc.Function.Arguments))
		b.WriteByte(0)
	}
	return b.String()
}

// sameStepsBefore counts the steps of the current turn (since the latest user
// message in the window) that immediately precede this one, made exactly the
// same calls, and got the same results as each other.
func sameStepsBefore(window []Message, calls []ToolCall) int {
	if len(calls) == 0 {
		return 0
	}
	sig := stepSignature(calls)
	results := map[string]string{} // tool call id → result
	var steps [][]ToolCall
	for _, m := range window {
		d := decodeMessage(m.Raw)
		switch {
		case d.Role == "user" && m.Kind == "":
			steps = steps[:0] // a new turn
		case d.Role == "assistant":
			steps = append(steps, d.ToolCalls)
		case d.Role == "tool":
			results[d.ToolCallID] = d.Content
		}
	}
	n, first := 0, ""
	for i := len(steps) - 1; i >= 0; i-- {
		if stepSignature(steps[i]) != sig {
			break
		}
		var r strings.Builder
		for _, tc := range steps[i] {
			r.WriteString(results[tc.ID])
			r.WriteByte(0)
		}
		if n > 0 && r.String() != first {
			break
		}
		first = r.String()
		n++
	}
	return n
}

func noProgressResult(times int) *RPCError {
	return &RPCError{Code: CodeCancelled, Message: fmt.Sprintf(
		"Not run: you have now made exactly this call %d times in a row, and it gave the same result each time. "+
			"The run was stopped so it does not keep spending. Next time, do something different: the call that does what you said you would do.", times)}
}

// onlyReadDoc reports a step that did nothing but read documents.
func onlyReadDoc(calls []RPCCall) bool {
	for _, c := range calls {
		if c.Method != ReadDocTool {
			return false
		}
	}
	return len(calls) > 0
}
