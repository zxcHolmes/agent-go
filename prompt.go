package agent

import (
	"encoding/json"
	"fmt"
	"strings"
)

// buildTool returns the single JSON-RPC tool definition. Output is
// deterministic so the request prefix stays cacheable.
func (a *Agent) buildTool() (json.RawMessage, error) {
	params := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"jsonrpc": map[string]any{"type": "string", "enum": []string{"2.0"}},
			"method":  map[string]any{"type": "string", "enum": a.methodNames(), "description": "RPC method name"},
			"params":  map[string]any{"type": "object", "description": "Method parameters"},
			"id":      map[string]any{"type": "integer", "description": "Request id, echoed in the response"},
		},
		"required": []string{"jsonrpc", "method", "params", "id"},
	}
	return marshalJSON(map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        a.cfg.ToolName,
			"description": "Call a backend method with one JSON-RPC 2.0 request. Returns a JSON-RPC response; on \"error\", read the message, fix the request and retry if appropriate.",
			"parameters":  params,
		},
	})
}

// buildSystemPrompt appends the RPC documentation to the configured prompt.
func (a *Agent) buildSystemPrompt() (string, error) {
	var b strings.Builder
	fileP := ""
	if a.res != nil {
		fileP = a.res.systemPrompt
	}
	for _, part := range []string{fileP, a.cfg.SystemPrompt, a.docsPromptSection()} {
		if part = strings.TrimRight(part, "\n"); part == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString("\n\n")
		}
		b.WriteString(part)
	}
	if len(a.methods) == 0 {
		return b.String(), nil
	}
	if b.Len() > 0 {
		b.WriteString("\n\n")
	}
	fmt.Fprintf(&b, "# Tool `%s`\n", a.cfg.ToolName)
	fmt.Fprintf(&b, "Use the `%s` tool to call backend methods. Its arguments are one JSON-RPC 2.0 request, e.g. {\"jsonrpc\":\"2.0\",\"method\":\"<name>\",\"params\":{...},\"id\":1}. ", a.cfg.ToolName)
	b.WriteString("The result is a JSON-RPC response. If it contains \"error\", read the message and correct your request. ")
	b.WriteString("Methods marked (requires confirmation) run only after the user approves; a rejection comes back as an error.")
	unlisted := false
	for _, m := range a.methods {
		unlisted = unlisted || m.Unlisted
	}
	if unlisted {
		b.WriteString(" More methods than the ones below exist; the documents that need them describe them, and they are called through this tool the same way.")
	}
	b.WriteString("\n\n## Methods\n")
	for _, name := range a.methodNames() {
		m := a.methods[name]
		if m.Unlisted {
			continue
		}
		if err := writeMethodDoc(&b, name, m); err != nil {
			return "", err
		}
	}
	if doc := strings.TrimSpace(a.cfg.RPCDoc); doc != "" {
		b.WriteString("\n## Documentation\n")
		b.WriteString(doc)
		b.WriteString("\n")
	}
	return b.String(), nil
}

// indent indents continuation lines so multi-line text stays under its method.
func indent(s string) string {
	return strings.ReplaceAll(s, "\n", "\n  ")
}

// MethodsDoc renders methods the way the system prompt lists them — name,
// description, params schema, result, examples, doc — in the order given. It
// is for documenting Unlisted methods elsewhere (e.g. inside a mounted doc), so
// their description can never drift from their definition. Listed methods may
// be passed too; the output does not depend on the flag.
func MethodsDoc(methods ...Method) (string, error) {
	var b strings.Builder
	for _, m := range methods {
		if err := writeMethodDoc(&b, m.Name, m); err != nil {
			return "", err
		}
	}
	return b.String(), nil
}

// writeMethodDoc writes one method's entry of the method list.
func writeMethodDoc(b *strings.Builder, name string, m Method) error {
	fmt.Fprintf(b, "- `%s`", name)
	if m.Description != "" {
		b.WriteString(": " + m.Description)
	}
	if m.RequireConfirm {
		b.WriteString(" (requires confirmation)")
	}
	b.WriteString("\n")
	if m.Params != nil {
		schema, err := marshalJSON(m.Params)
		if err != nil {
			return fmt.Errorf("agent: method %q params schema: %w", name, err)
		}
		fmt.Fprintf(b, "  params: %s\n", schema)
	}
	if r := strings.TrimSpace(m.Result); r != "" {
		fmt.Fprintf(b, "  returns: %s\n", indent(r))
	}
	for _, ex := range m.Examples {
		if !json.Valid([]byte(ex)) {
			return fmt.Errorf("agent: method %q example is not valid JSON: %s", name, ex)
		}
		fmt.Fprintf(b, "  example params: %s\n", ex)
	}
	if d := strings.TrimSpace(m.Doc); d != "" {
		fmt.Fprintf(b, "  %s\n", indent(d))
	}
	return nil
}
