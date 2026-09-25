// Package agent is a small agent SDK for OpenAI-compatible chat completion APIs.
//
// The model's main tool is a JSON-RPC 2.0 call. Every method the host
// application registers becomes callable through that tool, and the SDK
// dispatches each call to the matching Go handler. An optional built-in
// view_image tool lets the model look at images by URL. Output is streamed and
// saved to a SQL-like Store as it is generated, together with token usage,
// credits and RPC calls, so a session can be polled, resumed or recovered
// after a crash from its session id.
//
// Typical flow:
//
//	client, _ := agent.NewClient(ctx, agent.Config{    // once at startup
//		Store: agent.NewSQLStore(db, agent.SQLite),
//		BaseURL: "https://api.openai.com/v1", APIKey: key, Model: "gpt-4o-mini",
//		Methods: methods,
//	})
//	sid, _ := client.CreateSession(ctx, nil)
//	a, _ := client.Agent(ctx, sid, agent.AgentOptions{ContextParams: user})
//	res, _ := a.Chat(ctx, "hello")
//	fmt.Println(res.Reply())
package agent
