package examples

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

func TestViewImage(t *testing.T) {
	e := setup(t)
	e.cfg.ViewImage = true
	e.cfg.ViewImageDetail = "low"
	ctx := context.Background()
	a := e.newAgent(t, "")
	args, _ := json.Marshal(`{"url":"https://cdn.example.com/cat.png"}`)
	bad, _ := json.Marshal(`{"url":"ftp://x/y.png"}`)
	rpcArgs, _ := json.Marshal(`{"jsonrpc":"2.0","method":"get_order","params":{"order_id":"O"},"id":1}`)
	// One turn mixing a JSON-RPC call, a valid and an invalid view_image call.
	e.llm.push(fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[
		{"id":"t1","type":"function","function":{"name":"view_image","arguments":%s}},
		{"id":"t2","type":"function","function":{"name":"json_rpc","arguments":%s}},
		{"id":"t3","type":"function","function":{"name":"view_image","arguments":%s}}]}`, args, rpcArgs, bad),
		text("A cat."))
	res, err := a.Chat(ctx, "what is in this picture?")
	if err != nil || res.Reply() != "A cat." {
		t.Fatal(res, err)
	}
	if got := roles(res.Messages); got != "user,assistant,tool,tool,tool,user,assistant" {
		t.Fatal(got)
	}
	img := res.Messages[5]
	if img.Kind != agent.MessageKindViewImage || img.ToolCallID != "t1" || img.Status != agent.MessageDone {
		t.Fatalf("%+v", img)
	}
	if !strings.Contains(res.Messages[4].Content, `"ok":false`) {
		t.Fatalf("invalid url accepted: %s", res.Messages[4].Content)
	}

	// The tool is offered, and the model's next request carries the image part, unmodified.
	if !bytes.Contains(e.llm.requests[0]["tools"], []byte(`"name":"view_image"`)) {
		t.Fatal("view_image tool not offered")
	}
	msgs := e.llm.request(1)
	last := string(msgs[len(msgs)-1])
	if !strings.Contains(last, `{"type":"image_url","image_url":{"url":"https://cdn.example.com/cat.png","detail":"low"}}`) ||
		!strings.Contains(last, `{"role":"user","content":[{"type":"text"`) {
		t.Fatalf("image message: %s", last)
	}
	if !strings.Contains(string(msgs[len(msgs)-2]), `"tool_call_id":"t3"`) {
		t.Fatal("image must come after every tool result")
	}
}

// Seen live: the provider could not download the image URL and rejected the
// request; without healing, every later request replays the image and fails.
func TestUnloadableImageDoesNotPoisonSession(t *testing.T) {
	e := setup(t)
	e.cfg.ViewImage = true
	ctx := context.Background()
	a := e.newAgent(t, "")
	args, _ := json.Marshal(`{"url":"https://example.com/missing.png"}`)
	e.llm.push(fmt.Sprintf(`{"role":"assistant","content":null,"tool_calls":[{"id":"t1","type":"function","function":{"name":"view_image","arguments":%s}}]}`, args))
	e.llm.pushReply(reply{httpErr: 400, body: `{"error":{"message":"Error while downloading file. Upstream status code: 400.","param":"url"}}`})
	e.llm.push(text("I could not load that image."))
	res, err := a.Chat(ctx, "look at it")
	if err != nil || res.Reply() != "I could not load that image." {
		t.Fatal(res, err)
	}
	ms, _ := e.client.LatestMessages(ctx, a.SessionID(), 10)
	if statuses(ms) != "done,done,done,excluded,done" || !strings.Contains(ms[2].Content, "could not load this image") {
		t.Fatalf("%s %s", statuses(ms), ms[2].Content)
	}
	if strings.Contains(fmt.Sprint(e.llm.request(2)), "image_url") {
		t.Fatal("excluded image was sent again")
	}
	// Later turns keep working.
	e.llm.push(text("fine"))
	if res, err := a.Chat(ctx, "ok?"); err != nil || res.Reply() != "fine" {
		t.Fatal(res, err)
	}
	if strings.Contains(fmt.Sprint(e.llm.request(3)), "image_url") {
		t.Fatal("excluded image was sent again")
	}
}
