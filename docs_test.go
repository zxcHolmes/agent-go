package agent

import (
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
