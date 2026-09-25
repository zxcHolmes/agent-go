package examples

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"testing/fstest"

	agent "github.com/zxcHolmes/agent-go"
)

func readDocCall(id, args string) string {
	a, _ := json.Marshal(args)
	return fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":%q,"type":"function","function":{"name":"read_doc","arguments":%s}}]}`, id, a)
}

func TestMountedDocs(t *testing.T) {
	e := setup(t)
	e.cfg.SystemPrompt = "Answer in one sentence."
	e.cfg.SystemPromptFile = "prompts/system.md"
	e.cfg.DocPageChars = 60
	e.cfg.Docs = fstest.MapFS{
		"prompts/system.md": {Data: []byte("---\ntitle: core\n---\nYou are the ACME support agent.\n")},
		"refunds.md":        {Data: []byte("---\ntitle: Refund policy\ndescription: When and how refunds are paid\naudience: customers\n---\nRefunds are paid within 14 days.\n\nPartial refunds need a manager.\n\nGift cards are never refunded, no exceptions at all.")},
		"shipping/eu.md":    {Data: []byte("---\ntitle: EU shipping\ndescription: Delivery times in the EU\n---\n3-5 days.")},
	}
	ctx := context.Background()
	client, err := agent.NewClient(ctx, e.cfg)
	if err != nil {
		t.Fatal(err)
	}
	if docs := client.Docs(); len(docs) != 2 || docs[0].Title != "EU shipping" || docs[1].Meta["audience"] != "customers" {
		t.Fatalf("%+v", docs)
	}
	a, err := client.Agent(ctx, "", agent.AgentOptions{})
	if err != nil {
		t.Fatal(err)
	}
	e.llm.push(
		readDocCall("d1", `{"title":"refund policy"}`),          // case-insensitive
		readDocCall("d2", `{"title":"Refund policy","page":2}`), // second page
		readDocCall("d3", `{"title":"Warranty"}`),               // unknown
		text("Refunds take 14 days; gift cards are never refunded."),
	)
	res, err := a.Chat(ctx, "How do refunds work?")
	if err != nil {
		t.Fatal(err)
	}

	// System prompt: file prompt, then SystemPrompt, then the document index.
	var sys struct{ Content string }
	_ = json.Unmarshal(e.llm.request(0)[0], &sys)
	for _, want := range []string{
		"You are the ACME support agent.\n\nAnswer in one sentence.\n\n# Documents",
		"- `EU shipping`: Delivery times in the EU\n- `Refund policy`: When and how refunds are paid (audience: customers)",
	} {
		if !strings.Contains(sys.Content, want) {
			t.Fatalf("system prompt missing %q:\n%s", want, sys.Content)
		}
	}
	if strings.Contains(sys.Content, "`core`") {
		t.Fatal("the system prompt file must not be listed as a document")
	}
	if !strings.Contains(string(e.llm.requests[0]["tools"]), `"name":"read_doc"`) {
		t.Fatal("read_doc tool not offered")
	}

	var results []string
	for _, m := range res.Messages {
		if m.Role == "tool" {
			results = append(results, m.Content)
		}
	}
	if !strings.HasPrefix(results[0], "# Refund policy\n(page 1 of 3; call read_doc with {\"title\": \"Refund policy\", \"page\": 2} for the next page)\n\nRefunds are paid within 14 days.") ||
		strings.Contains(results[0], "---") {
		t.Fatalf("page 1 (plain text, no frontmatter):\n%s", results[0])
	}
	// Pages break at paragraph boundaries.
	if !strings.Contains(results[1], "(page 2 of 3;") || !strings.HasSuffix(results[1], "\n\nPartial refunds need a manager.") {
		t.Fatalf("page 2:\n%s", results[1])
	}
	if !strings.HasPrefix(results[2], `Error: no document titled "Warranty"`) {
		t.Fatalf("unknown title:\n%s", results[2])
	}

	// Without Docs there is no read_doc tool.
	e.cfg.Docs, e.cfg.SystemPromptFile = nil, ""
	plain := e.newAgent(t, "")
	e.llm.push(text("hi"))
	_, _ = plain.Chat(ctx, "hi")
	if strings.Contains(string(e.llm.requests[len(e.llm.requests)-1]["tools"]), "read_doc") {
		t.Fatal("read_doc offered without docs")
	}
}

func TestDocsLoadErrorsFailFast(t *testing.T) {
	e := setup(t)
	e.cfg.Docs = fstest.MapFS{
		"a.md": {Data: []byte("---\ntitle: Same\n---\n")},
		"b.md": {Data: []byte("---\ntitle: Same\n---\n")},
	}
	if _, err := agent.NewClient(context.Background(), e.cfg); err == nil || !strings.Contains(err.Error(), "unique") {
		t.Fatalf("want duplicate title error, got %v", err)
	}
	e.cfg.Docs = nil
	e.cfg.SystemPromptFile = "/no/such/file.md"
	if _, err := agent.NewClient(context.Background(), e.cfg); err == nil {
		t.Fatal("missing SystemPromptFile must fail")
	}
}
