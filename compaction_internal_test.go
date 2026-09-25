package agent

import (
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
