package agent

import (
	"context"
	"strings"
	"testing"
	"testing/fstest"
)

func TestParseFrontmatter(t *testing.T) {
	meta, body := parseFrontmatter("\ufeff---\r\ntitle: \"Refund policy\"\r\ndescription: >\r\n  How refunds\r\n  work.\r\nTags: billing, faq\r\n---\r\n# Body\r\ntext")
	if meta["title"] != "Refund policy" || meta["description"] != "How refunds work." || meta["tags"] != "billing, faq" {
		t.Fatalf("%q", meta)
	}
	if !strings.HasPrefix(body, "# Body") {
		t.Fatalf("body %q", body)
	}
	if meta, body := parseFrontmatter("# no frontmatter"); len(meta) != 0 || body != "# no frontmatter" {
		t.Fatal(meta, body)
	}
	if meta, _ := parseFrontmatter("---\ntitle: x\nnever closed"); len(meta) != 0 {
		t.Fatal("unterminated frontmatter must be ignored")
	}
}

func TestSplitPages(t *testing.T) {
	text := strings.Repeat("第一段内容。", 10) + "\n\n" + strings.Repeat("第二段内容。", 10)
	pages := splitPages(text, 80)
	if len(pages) != 2 || !strings.HasPrefix(pages[1], "第二段") {
		t.Fatalf("%d pages: %q", len(pages), pages)
	}
	if got := splitPages("short", 80); len(got) != 1 {
		t.Fatal(got)
	}
}

func TestLoadResources(t *testing.T) {
	fsys := fstest.MapFS{
		"system.md":      {Data: []byte("---\ntitle: ignored\n---\nYou are the support bot.")},
		"a.md":           {Data: []byte("---\ntitle: Alpha\ndescription: first\n---\nA")},
		"sub/b.markdown": {Data: []byte("---\ntitle: Beta\n---\nB")},
		"notes.txt":      {Data: []byte("not markdown")},
	}
	r, err := loadResources(Config{Docs: fsys, SystemPromptFile: "system.md"})
	if err != nil {
		t.Fatal(err)
	}
	if r.systemPrompt != "You are the support bot." || len(r.docs) != 2 || r.docs[0].Title != "Alpha" || r.docs[1].Path != "sub/b.markdown" {
		t.Fatalf("%+v", r)
	}
	fsys["dup.md"] = &fstest.MapFile{Data: []byte("---\ntitle: alpha\n---\n")}
	if _, err := loadResources(Config{Docs: fsys, SystemPromptFile: "system.md"}); err == nil || !strings.Contains(err.Error(), "unique") {
		t.Fatalf("want duplicate-title error, got %v", err)
	}
	delete(fsys, "dup.md")
	fsys["untitled.md"] = &fstest.MapFile{Data: []byte("# no frontmatter")}
	if _, err := loadResources(Config{Docs: fsys}); err == nil || !strings.Contains(err.Error(), "untitled.md") {
		t.Fatalf("want missing-title error, got %v", err)
	}
}

// A document already returned in the context window, unchanged, is answered
// with a short note; a changed one is sent again, marked as replacing the old
// copy; after compaction (the result no longer in the window) it is sent in full.
func TestReadDocSkipsWhatIsAlreadyLoaded(t *testing.T) {
	fsys := fstest.MapFS{"a.md": {Data: []byte("---\ntitle: Alpha\n---\nAlpha body")}}
	cfg := Config{Docs: fsys}
	res, err := loadResources(cfg)
	if err != nil {
		t.Fatal(err)
	}
	a := &Agent{log: newLogger(Config{}), cfg: cfg, res: res}
	read := func(window []Message, args string) string {
		m := &Message{ID: "m"}
		tc := ToolCall{ID: "now"}
		tc.Function.Name, tc.Function.Arguments = ReadDocTool, args
		c := a.newCall(context.Background(), m, 0, tc, a.loadedDocsIn(window))
		return toolContent(c.Result)
	}
	// window builds an assistant message calling read_doc and its tool answer.
	window := func(args, content string) []Message {
		tc := ToolCall{ID: "t1", Type: "function"}
		tc.Function.Name, tc.Function.Arguments = ReadDocTool, args
		asst, _ := marshalJSON(map[string]any{"role": "assistant", "content": "", "tool_calls": []ToolCall{tc}})
		return []Message{{Raw: asst}, {Raw: toolMessageRaw("t1", content)}}
	}

	full := read(nil, `{"title":"Alpha"}`)
	if !strings.Contains(full, "Alpha body") {
		t.Fatalf("first read: %q", full)
	}
	if got := read(window(`{"title":"alpha"}`, full), `{"title":"Alpha"}`); !strings.Contains(got, docAlreadyLoaded) || strings.Contains(got, "Alpha body") {
		t.Fatalf("unchanged document must not be sent again: %q", got)
	}
	// Compacted away (or never read): the window is empty, so it comes in full.
	if got := read(nil, `{"title":"Alpha"}`); got != full {
		t.Fatalf("not in the window: %q", got)
	}
	// Text that looks like a result but is not a read_doc answer does not count.
	if got := read([]Message{{Raw: toolMessageRaw("x", full)}}, `{"title":"Alpha"}`); got != full {
		t.Fatalf("an unmatched tool message must not count: %q", got)
	}

	// The document changes on disk: sent again, marked as the current version.
	fsys["a.md"] = &fstest.MapFile{Data: []byte("---\ntitle: Alpha\n---\nAlpha body v2")}
	changed := read(window(`{"title":"Alpha"}`, full), `{"title":"Alpha"}`)
	if !strings.Contains(changed, docChanged) || !strings.Contains(changed, "Alpha body v2") {
		t.Fatalf("changed document: %q", changed)
	}
	// That marked copy counts as loaded: reading again is a note, not a third copy.
	if got := read(window(`{"title":"Alpha"}`, changed), `{"title":"Alpha"}`); !strings.Contains(got, docAlreadyLoaded) {
		t.Fatalf("after the changed copy: %q", got)
	}
	// A note alone is not a copy: if only the note is left in the window, send it in full.
	note := read(window(`{"title":"Alpha"}`, changed), `{"title":"Alpha"}`)
	if got := read(window(`{"title":"Alpha"}`, note), `{"title":"Alpha"}`); !strings.Contains(got, "Alpha body v2") || strings.Contains(got, docChanged) {
		t.Fatalf("only a note in the window: %q", got)
	}

	// Two reads of the same document in one turn: the second is a note.
	loaded := a.loadedDocsIn(nil)
	first := a.readDoc(`{"title":"Alpha"}`, loaded)
	if second := a.readDoc(`{"title":"Alpha"}`, loaded); !strings.Contains(first, "Alpha body v2") || !strings.Contains(second, docAlreadyLoaded) {
		t.Fatalf("same turn: %q / %q", first, second)
	}
	// nil turns it off (crash recovery).
	if got := a.readDoc(`{"title":"Alpha"}`, nil); !strings.Contains(got, "Alpha body v2") {
		t.Fatalf("nil loaded: %q", got)
	}
}
