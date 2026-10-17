package agent

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestMessageTokensImagesAreFlat(t *testing.T) {
	short := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://a.b/c.png"}}]}`)
	long := json.RawMessage(`{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"https://a.b/` + strings.Repeat("x", 3000) + `.png"}}]}`)
	if s, l := messageTokens(short), messageTokens(long); s != l || s < imageTokens || s > imageTokens+50 {
		t.Fatalf("short %d, long %d", s, l)
	}
	two := json.RawMessage(`{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://a.b/1.png"}},{"type":"image_url","image_url":{"url":"https://a.b/2.png"}}]}`)
	if n := messageTokens(two); n < 2*imageTokens || n > 2*imageTokens+50 {
		t.Fatal(n)
	}
	text := json.RawMessage(`{"role":"user","content":"` + strings.Repeat("a", 300) + `"}`)
	if n := messageTokens(text); n != len(text)/3+4 {
		t.Fatal(n)
	}
}

// stepAnchor keeps the latest steps that fit in stepKeepShare of the
// threshold, always at least the latest one, and never cuts at a tool message
// or at the message that already starts the window.
func TestStepAnchor(t *testing.T) {
	a := &Agent{cfg: Config{ContextLength: 10000, CompactionThreshold: 0.8}} // keeps up to 2000 tokens
	msg := func(seq int64, role string, size int) Message {
		return Message{Seq: seq, Role: role, Raw: json.RawMessage(`{"c":"` + strings.Repeat("x", size) + `"}`)}
	}
	window := []Message{
		msg(1, "user", 10),
		msg(2, "assistant", 10), msg(3, "tool", 3000), // ~1000 tokens
		msg(4, "assistant", 10), msg(5, "tool", 3000), // ~1000 tokens
		msg(6, "assistant", 10), msg(7, "tool", 1500), // ~500 tokens
	}
	if got := a.stepAnchor(window, 1); got == nil || got.Seq != 4 {
		t.Fatalf("want the cut at #4 (the last two steps fit), got %+v", got)
	}
	huge := append(window[:5:5], msg(6, "assistant", 10), msg(7, "tool", 30000))
	if got := a.stepAnchor(huge, 1); got == nil || got.Seq != 6 {
		t.Fatalf("the latest step stays however big it is, got %+v", got)
	}
	if got := a.stepAnchor(window[:3], 2); got != nil {
		t.Fatalf("the only step already starts the window, got %+v", got)
	}
}
