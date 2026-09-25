package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime/debug"
	"sort"
	"strings"
)

// JSON-RPC error codes returned to the model.
const (
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternalError  = -32603
	CodeRejected       = -32001 // the user rejected a call that required confirmation
	CodeCancelled      = -32002 // the run was stopped before the call executed
	CodeCrashed        = -32003 // the process crashed while the call was running
)

// Handler executes one RPC method. The returned value is JSON-encoded into the
// "result" of the JSON-RPC response. Return an *RPCError (e.g. InvalidParams)
// to control the error code; any other error becomes an internal error whose
// message is shown to the model.
type Handler func(ctx context.Context, call *Call) (any, error)

// Method is an RPC method exposed to the model. NewMethod builds one from a
// typed function and derives Params from the params struct automatically.
type Method struct {
	Name string
	// Description is a one-line summary.
	Description string
	// Doc is a longer, free-form explanation: business rules, call order,
	// caveats. It may span several lines.
	Doc string
	// Params documents the params as a JSON Schema (map, struct or
	// json.RawMessage). It is shown to the model; validation is up to Handler.
	// ParamsSchema generates one from a struct.
	Params any
	// Result describes what the method returns.
	Result string
	// Examples are example params, each a JSON value such as `{"id":"A-1"}`.
	Examples []string
	// RequireConfirm pauses the agent before running this method until the
	// caller approves or rejects it with Agent.Confirm.
	RequireConfirm bool
	Handler        Handler
}

// MethodDoc is the documentation part of a Method, used by NewMethod.
type MethodDoc struct {
	Description    string
	Doc            string
	Result         string
	Examples       []string
	RequireConfirm bool
}

// NewMethod builds a Method from a typed function. Params are decoded and
// validated as with Typed, and the params JSON Schema shown to the model is
// generated from P's fields and struct tags (see ParamsSchema).
func NewMethod[P any, R any](name string, fn func(ctx context.Context, call *Call, params P) (R, error), doc MethodDoc) Method {
	return Method{
		Name:           name,
		Description:    doc.Description,
		Doc:            doc.Doc,
		Params:         ParamsSchema[P](),
		Result:         doc.Result,
		Examples:       doc.Examples,
		RequireConfirm: doc.RequireConfirm,
		Handler:        Typed(fn),
	}
}

// Call is the invocation context passed to a Handler.
type Call struct {
	SessionID  string
	CallID     string // RPCCall.ID
	ToolCallID string
	Method     string
	Params     json.RawMessage
	ID         json.RawMessage // JSON-RPC request id
	// ContextParams are the values given in AgentOptions.ContextParams, e.g. the
	// authenticated user. They are never sent to the model.
	ContextParams map[string]any
}

// Value returns a context param.
func (c *Call) Value(key string) any { return c.ContextParams[key] }

// Bind strictly decodes Params into v. Unknown fields are rejected so the
// model learns about typos. Errors are InvalidParams errors.
func (c *Call) Bind(v any) error {
	p := bytes.TrimSpace(c.Params)
	if len(p) == 0 || bytes.Equal(p, []byte("null")) {
		p = []byte("{}")
	}
	dec := json.NewDecoder(bytes.NewReader(p))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return InvalidParams("invalid params: %v", err)
	}
	return nil
}

type callKey struct{}

// CallFromContext returns the Call being handled, if any.
func CallFromContext(ctx context.Context) (*Call, bool) {
	c, ok := ctx.Value(callKey{}).(*Call)
	return c, ok
}

// Typed adapts a strongly typed function into a Handler. Params are decoded
// with Call.Bind; if P (or *P) has a `Validate() error` method it is called and
// its error is returned to the model as invalid params.
func Typed[P any, R any](fn func(ctx context.Context, call *Call, params P) (R, error)) Handler {
	return func(ctx context.Context, call *Call) (any, error) {
		var p P
		if err := call.Bind(&p); err != nil {
			return nil, err
		}
		if v, ok := any(&p).(interface{ Validate() error }); ok {
			if err := v.Validate(); err != nil {
				return nil, asInvalidParams(err)
			}
		} else if v, ok := any(p).(interface{ Validate() error }); ok {
			if err := v.Validate(); err != nil {
				return nil, asInvalidParams(err)
			}
		}
		return fn(ctx, call, p)
	}
}

func asInvalidParams(err error) error {
	var re *RPCError
	if errors.As(err, &re) {
		return re
	}
	return InvalidParams("%s", err.Error())
}

// RPCError is a JSON-RPC error object.
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

func (e *RPCError) Error() string { return fmt.Sprintf("rpc error %d: %s", e.Code, e.Message) }

// NewRPCError builds an RPCError with a custom code.
func NewRPCError(code int, message string, data any) *RPCError {
	return &RPCError{Code: code, Message: message, Data: data}
}

// InvalidParams builds a -32602 error. Handlers should return it when params fail validation.
func InvalidParams(format string, args ...any) *RPCError {
	return &RPCError{Code: CodeInvalidParams, Message: fmt.Sprintf(format, args...)}
}

// rpcRequest is the argument schema of the single tool given to the model.
type rpcRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params"`
	ID      json.RawMessage `json:"id"`
}

func parseRPCRequest(arguments string) (rpcRequest, *RPCError) {
	var req rpcRequest
	if err := json.Unmarshal([]byte(arguments), &req); err != nil {
		return req, &RPCError{Code: CodeParseError, Message: "tool arguments must be one JSON-RPC request object: " + err.Error()}
	}
	req.Params = bytes.TrimSpace(req.Params)
	// Some models send params as a JSON-encoded string.
	if len(req.Params) > 0 && req.Params[0] == '"' {
		var s string
		if json.Unmarshal(req.Params, &s) == nil && json.Valid([]byte(s)) {
			req.Params = json.RawMessage(s)
		}
	}
	if len(req.Params) == 0 || bytes.Equal(req.Params, []byte("null")) {
		req.Params = json.RawMessage("{}")
	}
	if req.Method == "" {
		return req, &RPCError{Code: CodeInvalidRequest, Message: `"method" is required`}
	}
	return req, nil
}

func rpcErrorResponse(id json.RawMessage, e *RPCError) json.RawMessage {
	if len(id) == 0 {
		id = json.RawMessage("null")
	}
	b, _ := marshalJSON(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Error   *RPCError       `json:"error"`
	}{"2.0", id, e})
	return b
}

func rpcResultResponse(id json.RawMessage, result any) (json.RawMessage, error) {
	if len(id) == 0 {
		id = json.RawMessage("null")
	}
	res, err := marshalJSON(result)
	if err != nil {
		return nil, err
	}
	return marshalJSON(struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  json.RawMessage `json:"result"`
	}{"2.0", id, res})
}

// invoke runs the handler for a call and returns the JSON-RPC response.
func (a *Agent) invoke(ctx context.Context, rec *RPCCall) (resp json.RawMessage) {
	m, ok := a.methods[rec.Method]
	if !ok {
		return rpcErrorResponse(rec.RPCID, a.methodNotFound(rec.Method))
	}
	call := &Call{
		SessionID:     rec.SessionID,
		CallID:        rec.ID,
		ToolCallID:    rec.ToolCallID,
		Method:        rec.Method,
		Params:        rec.Params,
		ID:            rec.RPCID,
		ContextParams: a.contextParams,
	}
	defer func() {
		if r := recover(); r != nil {
			resp = rpcErrorResponse(rec.RPCID, &RPCError{Code: CodeInternalError, Message: fmt.Sprintf("panic: %v", r), Data: string(debug.Stack())})
		}
	}()
	result, err := m.Handler(context.WithValue(ctx, callKey{}, call), call)
	if err != nil {
		var re *RPCError
		if !errors.As(err, &re) {
			re = &RPCError{Code: CodeInternalError, Message: err.Error()}
		}
		return rpcErrorResponse(rec.RPCID, re)
	}
	out, err := rpcResultResponse(rec.RPCID, result)
	if err != nil {
		return rpcErrorResponse(rec.RPCID, &RPCError{Code: CodeInternalError, Message: "encode result: " + err.Error()})
	}
	return out
}

func (a *Agent) methodNotFound(name string) *RPCError {
	return &RPCError{Code: CodeMethodNotFound, Message: fmt.Sprintf("method %q not found; available methods: %s", name, strings.Join(a.methodNames(), ", "))}
}

func (a *Agent) methodNames() []string {
	names := make([]string, 0, len(a.methods))
	for n := range a.methods {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

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
	b.WriteString(a.cfg.SystemPrompt)
	if len(a.methods) == 0 {
		return b.String(), nil
	}
	if b.Len() > 0 {
		b.WriteString("\n\n")
	}
	fmt.Fprintf(&b, "# Tool `%s`\n", a.cfg.ToolName)
	fmt.Fprintf(&b, "Use the `%s` tool to call backend methods. Its arguments are one JSON-RPC 2.0 request, e.g. {\"jsonrpc\":\"2.0\",\"method\":\"<name>\",\"params\":{...},\"id\":1}. ", a.cfg.ToolName)
	b.WriteString("The result is a JSON-RPC response. If it contains \"error\", read the message and correct your request. ")
	b.WriteString("Methods marked (requires confirmation) run only after the user approves; a rejection comes back as an error.\n\n## Methods\n")
	for _, name := range a.methodNames() {
		m := a.methods[name]
		fmt.Fprintf(&b, "- `%s`", name)
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
				return "", fmt.Errorf("agent: method %q params schema: %w", name, err)
			}
			fmt.Fprintf(&b, "  params: %s\n", schema)
		}
		if r := strings.TrimSpace(m.Result); r != "" {
			fmt.Fprintf(&b, "  returns: %s\n", indent(r))
		}
		for _, ex := range m.Examples {
			if !json.Valid([]byte(ex)) {
				return "", fmt.Errorf("agent: method %q example is not valid JSON: %s", name, ex)
			}
			fmt.Fprintf(&b, "  example params: %s\n", ex)
		}
		if d := strings.TrimSpace(m.Doc); d != "" {
			fmt.Fprintf(&b, "  %s\n", indent(d))
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
