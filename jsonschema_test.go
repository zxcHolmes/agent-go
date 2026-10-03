package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"
)

type Base struct {
	TraceID string `json:"trace_id,omitempty" desc:"Trace id"`
}

type Item struct {
	SKU string `json:"sku" desc:"SKU code"`
	Qty int    `json:"qty"`
}

type Node struct {
	Name     string  `json:"name"`
	Children []*Node `json:"children,omitempty"`
}

type refundParams struct {
	Base
	OrderID  string            `json:"order_id" desc:"Order id, e.g. ORD-123"`
	Mode     string            `json:"mode" enum:"full,partial"`
	Level    int               `json:"level" enum:"1,2,3"`
	Amount   float64           `json:"amount,omitempty" desc:"Partial amount"`
	Items    []Item            `json:"items" required:"false"`
	Note     *string           `json:"note"`
	Tags     map[string]string `json:"tags,omitempty"`
	At       time.Time         `json:"at,omitempty"`
	Count    int64             `json:"count,string,omitempty"`
	Extra    any               `json:"extra,omitempty"`
	Tree     *Node             `json:"tree,omitempty"`
	Skipped  string            `json:"-"`
	internal string
}

func TestParamsSchema(t *testing.T) {
	got := string(ParamsSchema[refundParams]())
	want := `{"type":"object","properties":{` +
		`"trace_id":{"type":"string","description":"Trace id"},` +
		`"order_id":{"type":"string","description":"Order id, e.g. ORD-123"},` +
		`"mode":{"type":"string","enum":["full","partial"]},` +
		`"level":{"type":"integer","enum":[1,2,3]},` +
		`"amount":{"type":"number","description":"Partial amount"},` +
		`"items":{"type":"array","items":{"type":"object","properties":{"sku":{"type":"string","description":"SKU code"},"qty":{"type":"integer"}},"required":["sku","qty"]}},` +
		`"note":{"type":"string"},` +
		`"tags":{"type":"object","additionalProperties":{"type":"string"}},` +
		`"at":{"type":"string","format":"date-time"},` +
		`"count":{"type":"string"},` +
		`"extra":{},` +
		`"tree":{"type":"object","properties":{"name":{"type":"string"},"children":{"type":"array","items":{"type":"object"}}},"required":["name"]}` +
		`},"required":["order_id","mode","level"]}`
	if got != want {
		t.Fatalf("got\n%s\nwant\n%s", got, want)
	}
	if !json.Valid([]byte(got)) {
		t.Fatal("invalid json")
	}
	if s := string(ParamsSchema[struct{}]()); s != `{"type":"object"}` {
		t.Fatal(s)
	}
}

func TestNewMethodPrompt(t *testing.T) {
	m := NewMethod("refund_order", func(ctx context.Context, c *Call, p Item) (string, error) {
		return p.SKU, nil
	}, MethodDoc{
		Description:    "Refund an order",
		Doc:            "Call get_order first.\nPartial refunds need items.",
		Result:         `{"refunded": number}`,
		Examples:       []string{`{"sku":"A","qty":1}`},
		RequireConfirm: true,
	})
	a := &Agent{cfg: Config{ToolName: "json_rpc"}, methods: map[string]Method{m.Name: m}}
	p, err := a.buildSystemPrompt()
	if err != nil {
		t.Fatal(err)
	}
	wantPart := "- `refund_order`: Refund an order (requires confirmation)\n" +
		`  params: {"type":"object","properties":{"sku":{"type":"string","description":"SKU code"},"qty":{"type":"integer"}},"required":["sku","qty"]}` + "\n" +
		`  returns: {"refunded": number}` + "\n" +
		`  example params: {"sku":"A","qty":1}` + "\n" +
		"  Call get_order first.\n  Partial refunds need items.\n"
	if !strings.Contains(p, wantPart) {
		t.Fatalf("prompt:\n%s", p)
	}
	out, err := m.Handler(context.Background(), &Call{Params: json.RawMessage(`{"sku":"X","qty":2}`)})
	if err != nil || out.(string) != "X" {
		t.Fatal(out, err)
	}

	m.Examples = []string{`{bad`}
	a.methods[m.Name] = m
	if _, err := a.buildSystemPrompt(); err == nil {
		t.Fatal("want invalid example error")
	}
}

func TestUnlistedMethod(t *testing.T) {
	listed := NewMethod("get_order", func(ctx context.Context, c *Call, p Item) (string, error) { return p.SKU, nil },
		MethodDoc{Description: "Read an order"})
	hidden := NewMethod("refund_order", func(ctx context.Context, c *Call, p Item) (string, error) { return "refunded " + p.SKU, nil },
		MethodDoc{Description: "Refund an order", Unlisted: true, RequireConfirm: true})
	a := &Agent{cfg: Config{ToolName: "json_rpc"}, methods: map[string]Method{listed.Name: listed, hidden.Name: hidden}}
	p, err := a.buildSystemPrompt()
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(p, "- `get_order`") || strings.Contains(p, "refund_order") {
		t.Fatalf("listed must appear and unlisted must not:\n%s", p)
	}
	if !strings.Contains(p, "More methods than the ones below exist") {
		t.Fatalf("the prompt should say that more methods exist:\n%s", p)
	}
	// Still dispatched like any method.
	resp := a.invoke(context.Background(), &RPCCall{Method: "refund_order", Params: json.RawMessage(`{"sku":"X","qty":1}`), RPCID: json.RawMessage(`1`)})
	if !strings.Contains(string(resp), "refunded X") {
		t.Fatalf("unlisted method was not dispatched: %s", resp)
	}
	// MethodsDoc renders it the way the prompt would have.
	doc, err := MethodsDoc(hidden)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(doc, "- `refund_order`: Refund an order (requires confirmation)\n  params: {") {
		t.Fatalf("doc:\n%s", doc)
	}

	// Nothing unlisted: no extra sentence.
	delete(a.methods, hidden.Name)
	p, _ = a.buildSystemPrompt()
	if strings.Contains(p, "More methods") {
		t.Fatal("no unlisted methods, no hint")
	}
}
