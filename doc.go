// Package agent is a small agent SDK for OpenAI-compatible chat completion APIs.
//
// The model is given exactly one tool: a JSON-RPC 2.0 call. Every method the
// host application registers becomes callable through that tool, and the SDK
// dispatches each call to the matching Go handler. Conversations, token usage,
// credits and RPC calls are persisted through a SQL-like Store, so a session can
// be resumed at any time from its session id.
//
// Typical flow:
//
//	store := agent.NewSQLStore(db, agent.SQLite)
//	_ = agent.Init(ctx, store)                  // create tables once at startup
//	sid, _ := agent.CreateSession(ctx, store, nil)
//	a, _ := agent.New(ctx, agent.Config{Store: store, SessionID: sid, ...})
//	res, _ := a.Chat(ctx, "hello")
//	fmt.Println(res.Reply())
package agent
