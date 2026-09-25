package agent

import (
	"strings"
	"testing"
)

func TestViewImageCall(t *testing.T) {
	a := &Agent{cfg: Config{ToolName: "json_rpc", ViewImage: true}, methods: map[string]Method{}}
	m := &Message{ID: "m"}
	tc := ToolCall{ID: "t1"}
	tc.Function.Name = ViewImageTool

	tc.Function.Arguments = `{"url":"https://cdn.example.com/a.png"}`
	c := a.newCall(m, 0, tc)
	if u, ok := viewImageURL(&c); !ok || u != "https://cdn.example.com/a.png" || c.Status != CallDone {
		t.Fatalf("%+v", c)
	}
	for _, bad := range []string{`{"url":"file:///etc/passwd"}`, `{"url":""}`, `{"url":"data:image/png;base64,xx"}`, `not json`} {
		tc.Function.Arguments = bad
		c := a.newCall(m, 0, tc)
		if _, ok := viewImageURL(&c); ok || !strings.Contains(string(c.Result), `"ok":false`) {
			t.Fatalf("%s accepted: %s", bad, c.Result)
		}
	}

	// Disabled: view_image is an unknown tool.
	a.cfg.ViewImage = false
	tc.Function.Arguments = `{"url":"https://cdn.example.com/a.png"}`
	c = a.newCall(m, 0, tc)
	if !strings.Contains(string(c.Result), "unknown tool") {
		t.Fatalf("%s", c.Result)
	}
}
