package agent

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestTranscriptKeepsHeadAndTail(t *testing.T) {
	var ms []Message
	ms = append(ms, Message{Role: "user", Content: "my code word is PELICAN"})
	for i := 0; i < 30; i++ {
		ms = append(ms, Message{Role: "assistant", Content: strings.Repeat("长篇内容", 800)})
	}
	ms = append(ms, Message{Role: "user", Content: "latest question"})
	s := transcript(ms, 6000)
	if len(s) > 6000 || !utf8.ValidString(s) {
		t.Fatalf("len=%d valid=%v", len(s), utf8.ValidString(s))
	}
	if !strings.Contains(s, "PELICAN") || !strings.Contains(s, "latest question") || !strings.Contains(s, "omitted") {
		t.Fatalf("lost head or tail:\n%.300s … %.300s", s, s[len(s)-300:])
	}
}

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
