package agent

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"strings"
	"testing"
)

func TestParseUsage(t *testing.T) {
	u := parseUsage(json.RawMessage(`{"prompt_tokens":1000,"completion_tokens":200,"total_tokens":1200,
		"prompt_tokens_details":{"cached_tokens":600,"audio_tokens":3},
		"completion_tokens_details":{"reasoning_tokens":50,"accepted_prediction_tokens":1,"rejected_prediction_tokens":2}}`))
	want := Usage{PromptTokens: 1000, CompletionTokens: 200, TotalTokens: 1200, CachedTokens: 600, ReasoningTokens: 50,
		AudioInputTokens: 3, AcceptedPredictionTokens: 1, RejectedPredictionTokens: 2}
	if u != want {
		t.Fatalf("got %+v want %+v", u, want)
	}
	ds := parseUsage(json.RawMessage(`{"prompt_tokens":10,"completion_tokens":5,"prompt_cache_hit_tokens":4}`))
	if ds.CachedTokens != 4 || ds.TotalTokens != 15 {
		t.Fatalf("deepseek usage: %+v", ds)
	}
}

func TestPricing(t *testing.T) {
	p := Pricing{Input: 100, Output: 1000, CacheRead: 10}
	c := p.Cost("m", Usage{PromptTokens: 1_000_000, CachedTokens: 400_000, CompletionTokens: 100_000})
	if math.Abs(c.Input-60) > 1e-9 || math.Abs(c.CachedInput-4) > 1e-9 || math.Abs(c.Output-100) > 1e-9 || math.Abs(c.Total-164) > 1e-9 {
		t.Fatalf("cost %+v", c)
	}
	tbl := PricingTable{Default: Pricing{Input: 1}, Models: map[string]Pricing{"big": {Input: 2}}}
	if tbl.Cost("big", Usage{PromptTokens: 1e6}).Total != 2 || tbl.Cost("x", Usage{PromptTokens: 1e6}).Total != 1 {
		t.Fatal("pricing table")
	}
}

func TestParseRPCRequest(t *testing.T) {
	req, err := parseRPCRequest(`{"jsonrpc":"2.0","method":"a","params":"{\"x\":1}","id":3}`)
	if err != nil || req.Method != "a" || string(req.Params) != `{"x":1}` || string(req.ID) != "3" {
		t.Fatalf("%+v %v", req, err)
	}
	if _, err := parseRPCRequest(`not json`); err == nil || err.Code != CodeParseError {
		t.Fatalf("want parse error, got %v", err)
	}
	if _, err := parseRPCRequest(`{"params":{}}`); err == nil || err.Code != CodeInvalidRequest {
		t.Fatalf("want invalid request, got %v", err)
	}
}

type addParams struct {
	A, B int
}

func (p addParams) Validate() error {
	if p.B == 0 {
		return errors.New("b must not be zero")
	}
	return nil
}

func TestTypedHandler(t *testing.T) {
	h := Typed(func(ctx context.Context, c *Call, p addParams) (int, error) { return p.A / p.B, nil })
	got, err := h(context.Background(), &Call{Params: json.RawMessage(`{"A":6,"B":3}`)})
	if err != nil || got.(int) != 2 {
		t.Fatalf("got %v %v", got, err)
	}
	for _, params := range []string{`{"A":1,"B":0}`, `{"A":1,"C":2}`, `{"A":"x"}`} {
		_, err := h(context.Background(), &Call{Params: json.RawMessage(params)})
		var re *RPCError
		if !errors.As(err, &re) || re.Code != CodeInvalidParams {
			t.Fatalf("%s: want invalid params, got %v", params, err)
		}
	}
}

func TestInvoke(t *testing.T) {
	a := &Agent{cfg: Config{ContextParams: map[string]any{"user": "u1"}}, methods: map[string]Method{
		"whoami": {Name: "whoami", Handler: func(ctx context.Context, c *Call) (any, error) {
			cc, _ := CallFromContext(ctx)
			return map[string]any{"user": cc.Value("user")}, nil
		}},
		"boom": {Name: "boom", Handler: func(ctx context.Context, c *Call) (any, error) { panic("x") }},
	}}
	out := a.invoke(context.Background(), &RPCCall{Method: "whoami", RPCID: json.RawMessage("7")})
	if string(out) != `{"jsonrpc":"2.0","id":7,"result":{"user":"u1"}}` {
		t.Fatalf("got %s", out)
	}
	out = a.invoke(context.Background(), &RPCCall{Method: "boom", RPCID: json.RawMessage("1")})
	if !strings.Contains(string(out), `"code":-32603`) {
		t.Fatalf("panic not converted: %s", out)
	}
	out = a.invoke(context.Background(), &RPCCall{Method: "nope", RPCID: json.RawMessage("1")})
	if !strings.Contains(string(out), `"code":-32601`) || !strings.Contains(string(out), "boom, whoami") {
		t.Fatalf("method not found: %s", out)
	}
}

func TestRebindAndSchema(t *testing.T) {
	s := NewSQLStore(nil, Postgres)
	if got := s.rebind("a = ? AND b IN (?, ?)"); got != "a = $1 AND b IN ($2, $3)" {
		t.Fatal(got)
	}
	for _, d := range []Dialect{SQLite, Postgres, MySQL} {
		stmts := SchemaStatements(d)
		joined := strings.Join(stmts, "\n")
		if d == MySQL && (strings.Contains(joined, "CREATE INDEX") || !strings.Contains(joined, "LONGTEXT")) {
			t.Fatalf("mysql ddl:\n%s", joined)
		}
		if d != MySQL && !strings.Contains(joined, "CREATE INDEX IF NOT EXISTS") {
			t.Fatalf("%s ddl:\n%s", d, joined)
		}
	}
}

func TestBuildBodyExtra(t *testing.T) {
	req := chatRequest{Model: "m", Messages: []json.RawMessage{json.RawMessage(`{"role":"user","content":"<a&b>"}`)},
		Stream: true, StreamOptions: &streamOptions{IncludeUsage: true}}
	b, err := buildBody(req, map[string]any{"temperature": 0.2, "model": "hijack", "stream": false})
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != `{"messages":[{"role":"user","content":"<a&b>"}],"model":"m","stream":true,"stream_options":{"include_usage":true},"temperature":0.2}` {
		t.Fatalf("got %s", b)
	}
	b, _ = buildBody(req, map[string]any{"stream_options": nil})
	if strings.Contains(string(b), "stream_options") {
		t.Fatalf("nil should remove the field: %s", b)
	}
}
