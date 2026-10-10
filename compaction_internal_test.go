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
