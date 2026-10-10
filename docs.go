package agent

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path"
	"sort"
	"strings"
)

// ReadDocTool is the name of the built-in tool offered when Config.Docs is
// set. The model passes a document title (and optionally a page) and gets the
// document's markdown back as plain text.
const ReadDocTool = "read_doc"

// Doc describes one mounted markdown document. Its frontmatter is shown to
// the model in the system prompt; the body is only read when the model asks.
type Doc struct {
	Title       string            `json:"title"`
	Description string            `json:"description,omitempty"`
	Meta        map[string]string `json:"meta,omitempty"` // other frontmatter keys
	Path        string            `json:"path"`           // path inside Config.Docs
}

// resources are the file-backed parts of an agent's configuration, loaded
// once per Client (and again for agents whose config is overridden).
type resources struct {
	docs         []Doc // sorted by title
	byTitle      map[string]*Doc
	systemPrompt string // body of Config.SystemPromptFile
}

func loadResources(cfg Config) (*resources, error) {
	r := &resources{byTitle: map[string]*Doc{}}
	systemPath := ""
	if cfg.SystemPromptFile != "" {
		var data []byte
		var err error
		if cfg.Docs != nil {
			systemPath = path.Clean(cfg.SystemPromptFile)
			data, err = fs.ReadFile(cfg.Docs, systemPath)
		} else {
			data, err = os.ReadFile(cfg.SystemPromptFile)
		}
		if err != nil {
			return nil, fmt.Errorf("agent: read SystemPromptFile: %w", err)
		}
		_, body := parseFrontmatter(string(data))
		r.systemPrompt = strings.TrimSpace(body)
	}
	if cfg.Docs == nil {
		return r, nil
	}
	err := fs.WalkDir(cfg.Docs, ".", func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		ext := strings.ToLower(path.Ext(p))
		if d.IsDir() || (ext != ".md" && ext != ".markdown") || p == systemPath {
			return nil
		}
		data, err := fs.ReadFile(cfg.Docs, p)
		if err != nil {
			return err
		}
		meta, _ := parseFrontmatter(string(data))
		doc := Doc{Title: meta["title"], Description: meta["description"], Path: p}
		delete(meta, "title")
		delete(meta, "description")
		if len(meta) > 0 {
			doc.Meta = meta
		}
		if doc.Title == "" {
			return fmt.Errorf("agent: document %s has no \"title\" in its frontmatter", p)
		}
		key := strings.ToLower(doc.Title)
		if other, dup := r.byTitle[key]; dup {
			return fmt.Errorf("agent: documents %s and %s share the title %q; titles must be unique", other.Path, p, doc.Title)
		}
		r.docs = append(r.docs, doc)
		r.byTitle[key] = &r.docs[len(r.docs)-1]
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("agent: load docs: %w", err)
	}
	sort.Slice(r.docs, func(i, j int) bool { return r.docs[i].Title < r.docs[j].Title })
	for i := range r.docs { // re-point after sorting
		r.byTitle[strings.ToLower(r.docs[i].Title)] = &r.docs[i]
	}
	return r, nil
}

// parseFrontmatter splits a leading "---" block of "key: value" lines from
// the body. Values may be quoted; indented lines continue the previous value
// (enough for YAML "|" / ">" blocks). Keys are lower-cased.
func parseFrontmatter(s string) (map[string]string, string) {
	meta := map[string]string{}
	s = strings.TrimPrefix(s, "\ufeff")
	lines := strings.Split(s, "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return meta, s
	}
	key := ""
	for i := 1; i < len(lines); i++ {
		line := strings.TrimRight(lines[i], "\r")
		if strings.TrimSpace(line) == "---" {
			return meta, strings.Join(lines[i+1:], "\n")
		}
		if key != "" && (strings.HasPrefix(line, " ") || strings.HasPrefix(line, "\t")) {
			v := strings.TrimSpace(line)
			if meta[key] != "" && v != "" {
				v = meta[key] + " " + v
			} else if v == "" {
				v = meta[key]
			}
			meta[key] = v
			continue
		}
		k, v, ok := strings.Cut(line, ":")
		if !ok || strings.TrimSpace(k) == "" {
			continue
		}
		key = strings.ToLower(strings.TrimSpace(k))
		v = strings.TrimSpace(v)
		if v == "|" || v == ">" || v == "|-" || v == ">-" {
			v = ""
		}
		if len(v) >= 2 && (v[0] == '"' && v[len(v)-1] == '"' || v[0] == '\'' && v[len(v)-1] == '\'') {
			v = v[1 : len(v)-1]
		}
		meta[key] = v
	}
	return map[string]string{}, s // no closing "---": not frontmatter
}

func (a *Agent) buildReadDocTool() (json.RawMessage, error) {
	return marshalJSON(map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        ReadDocTool,
			"description": "Read one of the documents listed in the system prompt, by its exact title. Start with page 1 (the default): most documents have a single page, and the result says explicitly when there are more pages. A document you already read in this conversation is not sent again while it is unchanged: the result then says it is already loaded, and you use the earlier copy.",
			"parameters": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"title": map[string]any{"type": "string", "description": "Exact document title"},
					"page":  map[string]any{"type": "integer", "description": "Page number; omit for page 1. Only request later pages when the previous result says they exist."},
				},
				"required": []string{"title"},
			},
		},
	})
}

func (a *Agent) hasDocs() bool { return a.res != nil && len(a.res.docs) > 0 }

// docsPromptSection lists the mounted documents for the system prompt.
func (a *Agent) docsPromptSection() string {
	if a.res == nil || len(a.res.docs) == 0 {
		return ""
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Documents\nThese documents are available through the `%s` tool (pass the exact title). ", ReadDocTool)
	b.WriteString("Read the relevant document before answering questions it covers instead of guessing.\n\n")
	for _, d := range a.res.docs {
		fmt.Fprintf(&b, "- `%s`", d.Title)
		if d.Description != "" {
			b.WriteString(": " + d.Description)
		}
		if len(d.Meta) > 0 {
			keys := make([]string, 0, len(d.Meta))
			for k := range d.Meta {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			var kv []string
			for _, k := range keys {
				kv = append(kv, k+": "+d.Meta[k])
			}
			b.WriteString(" (" + strings.Join(kv, "; ") + ")")
		}
		b.WriteString("\n")
	}
	return b.String()
}

// loadedDocs is what the model can already see of the mounted documents: for
// each (title, page), the text of the latest read_doc result in the context
// window this step sends, plus the results answered earlier in the same turn.
// Compaction drops old results from the window, so a document read before a
// compaction is sent again in full.
type loadedDocs map[docPage]string

type docPage struct {
	title string // lower-cased
	page  int
}

// Lines read_doc puts right under a result's "# Title" heading.
const (
	docAlreadyLoaded = "(Already loaded: this document is unchanged since your earlier read_doc call, whose result is still in this conversation. Use that copy; it is not sent again.)"
	docChanged       = "(This document changed since you last read it. This is the current version; it replaces the earlier copy.)"
)

// loadedDocsIn collects the read_doc results in a context window. A result is
// matched to its call through the assistant message's tool calls, so only
// real read_doc answers count, never text that merely looks like one.
func (a *Agent) loadedDocsIn(window []Message) loadedDocs {
	l := loadedDocs{}
	if !a.hasDocs() {
		return l
	}
	asked := map[string]docPage{} // tool call id → what it asked for
	for _, m := range window {
		d := decodeMessage(m.Raw)
		switch d.Role {
		case "assistant":
			for _, tc := range d.ToolCalls {
				if tc.Function.Name != ReadDocTool {
					continue
				}
				if doc, page, ok := a.docArgs(tc.Function.Arguments); ok {
					asked[tc.ID] = docPage{strings.ToLower(doc.Title), page}
				}
			}
		case "tool":
			key, ok := asked[d.ToolCallID]
			if !ok || !strings.HasPrefix(d.Content, "# ") {
				continue // not a read_doc call, or an error answer
			}
			if _, rest, _ := strings.Cut(d.Content, "\n"); strings.HasPrefix(rest, docAlreadyLoaded) {
				continue // a pointer to an earlier copy, not a copy
			}
			l[key] = strings.Replace(d.Content, "\n"+docChanged, "", 1)
		}
	}
	return l
}

// newReadDocCall answers a read_doc call right away (reading a file needs no
// confirmation). The result is plain text, stored as a JSON string. With
// loaded (nil disables it), a document the model already has, unchanged, is
// answered with a short note instead of its text.
func (a *Agent) newReadDocCall(c *RPCCall, arguments string, loaded loadedDocs) {
	c.Method, c.Status = ReadDocTool, CallDone
	text := a.readDoc(arguments, loaded)
	c.Params = json.RawMessage(arguments)
	if !json.Valid(c.Params) {
		c.Params, _ = marshalJSON(arguments)
	}
	c.Result, _ = marshalJSON(text)
}

// docArgs resolves read_doc arguments to a mounted document and a page (1 when omitted).
func (a *Agent) docArgs(arguments string) (*Doc, int, bool) {
	var args struct {
		Title string `json:"title"`
		Page  int    `json:"page"`
	}
	if json.Unmarshal([]byte(arguments), &args) != nil {
		return nil, 0, false
	}
	doc := a.res.byTitle[strings.ToLower(strings.TrimSpace(args.Title))]
	if args.Page <= 0 {
		args.Page = 1
	}
	return doc, args.Page, doc != nil
}

func (a *Agent) readDoc(arguments string, loaded loadedDocs) string {
	var args struct {
		Title string `json:"title"`
		Page  int    `json:"page"`
	}
	if err := json.Unmarshal([]byte(arguments), &args); err != nil {
		return "Error: arguments must be a JSON object like {\"title\": \"...\"}: " + err.Error()
	}
	doc := a.res.byTitle[strings.ToLower(strings.TrimSpace(args.Title))]
	if doc == nil {
		return fmt.Sprintf("Error: no document titled %q. Use one of the exact titles listed in the system prompt.", args.Title)
	}
	data, err := fs.ReadFile(a.cfg.Docs, doc.Path)
	if err != nil {
		return "Error: could not read the document: " + err.Error()
	}
	_, body := parseFrontmatter(string(data))
	pages := splitPages(strings.TrimSpace(body), a.cfg.DocPageChars)
	page := args.Page
	if page <= 0 {
		page = 1
	}
	if page > len(pages) {
		return fmt.Sprintf("Error: %q has %d page(s); page %d does not exist.", doc.Title, len(pages), page)
	}
	head := "# " + doc.Title + "\n"
	if len(pages) > 1 {
		head += fmt.Sprintf("(page %d of %d", page, len(pages))
		if page < len(pages) {
			head += fmt.Sprintf("; call %s with {\"title\": %q, \"page\": %d} for the next page", ReadDocTool, doc.Title, page+1)
		}
		head += ")\n"
	}
	text := head + "\n" + pages[page-1]
	if loaded == nil {
		return text
	}
	key := docPage{strings.ToLower(doc.Title), page}
	prev, seen := loaded[key]
	loaded[key] = text
	switch {
	case !seen:
		return text
	case prev == text:
		return "# " + doc.Title + "\n" + docAlreadyLoaded
	default:
		return "# " + doc.Title + "\n" + docChanged + strings.TrimPrefix(text, "# "+doc.Title)
	}
}

// splitPages cuts text into pages of at most size characters, preferring to
// break at a blank line or newline.
func splitPages(text string, size int) []string {
	r := []rune(text)
	if size <= 0 || len(r) <= size {
		return []string{text}
	}
	var pages []string
	for len(r) > size {
		cut := size
		chunk := string(r[:size])
		if i := strings.LastIndex(chunk, "\n\n"); i > len(chunk)/2 {
			cut = len([]rune(chunk[:i]))
		} else if i := strings.LastIndex(chunk, "\n"); i > len(chunk)/2 {
			cut = len([]rune(chunk[:i]))
		}
		pages = append(pages, strings.TrimSpace(string(r[:cut])))
		r = r[cut:]
	}
	return append(pages, strings.TrimSpace(string(r)))
}

// toolContent is the text of a tool message for a call result: plain-text
// results (stored as a JSON string) are unquoted, JSON objects are kept.
func toolContent(result json.RawMessage) string {
	if len(result) > 0 && result[0] == '"' {
		var s string
		if json.Unmarshal(result, &s) == nil {
			return s
		}
	}
	return string(result)
}
