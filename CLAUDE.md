# agent-go

Go agent SDK for OpenAI-compatible chat APIs, imported by other projects as
`github.com/zxcHolmes/agent-go` (package `agent`, public repo). The model gets a
single JSON-RPC 2.0 tool (`json_rpc`) that dispatches to Go handlers registered by
the host app, plus built-in tools: `read_doc` (mounted markdown docs) and optional
`view_image`. Everything (sessions,
messages, RPC calls, token usage, billing, queued messages) is persisted through a
SQL-like `Store`, so sessions can be polled by a frontend, resumed by id and
recovered after crashes.

Main features: streaming with throttled DB flushes, confirmation-gated RPC methods,
non-blocking cross-process Stop, crash recovery, context compaction (anchor /
summary), user message queue, reminder, RPC result limit, credit billing with safe
settlement. User-facing docs are in `README.md` (Chinese).

## Tech stack & conventions

- Go 1.21 module, **zero third-party dependencies in the root module**. Anything
  needing a driver or library goes in `examples/` (its own module) or `demo/`.
- Public API goes through `*Client` (created once, holds `Config` + `Store`) and
  `*Agent` (per session, `client.Agent(ctx, sid, AgentOptions{...})`). Do not add
  package-level functions that take a `Store`; store helpers are unexported.
- Every public method takes `ctx context.Context` first. Never store a ctx.
- Portable SQL (SQLite / Postgres / MySQL), see `schema.go`:
  - write queries with `?` placeholders (rebound to `$n` for Postgres);
  - timestamps are unix **milliseconds** in `BIGINT`; ids are app-generated strings
    (`newID("msg")`), never AUTOINCREMENT / RETURNING;
  - all columns `NOT NULL`, always inserted explicitly (MySQL TEXT has no default);
  - MySQL indexes are declared inline (no `CREATE INDEX IF NOT EXISTS`);
  - don't trust RowsAffected (MySQL reports changed rows); read back instead
    (see `acquire`).
- Caller-built user messages (`ChatMessage`, `EnqueueMessage`) go through
  `checkUserMessage`: only `text` / `image_url` parts, images must be http(s)
  URLs (no base64). The token estimate counts each image as 500 tokens.
- JSON: use `marshalJSON` (no HTML escaping). A message's `raw` column is the exact
  bytes sent to the model and must not change once final — it is replayed
  byte-for-byte so provider prefix caches keep hitting. Build messages with
  structs (not maps) when key order matters for readability.
- Built-in tools (`read_doc`, `view_image`) are answered at call creation
  (`newCall`), never executed by the loop; plain-text results are stored as a JSON
  string and unquoted into the tool message (`toolContent`).
- Errors shown to the model are JSON-RPC errors (`rpc.go` codes: -32602 invalid
  params, -32001 rejected, -32002 cancelled/stopped, -32003 crashed). Handler
  errors never abort the loop.
- Comments/docs in English in code; README in Chinese.
- The schema can still change freely (no users yet): edit `schema.go` directly, no
  migrations.

## Invariants the run loop relies on

- Every assistant `tool_calls` entry has exactly one tool message, written right
  after it (created `pending` together with the call, updated in place). Injected
  `view_image` user messages go after all tool messages of the turn.
- A message is final when `Status.Final()`; only final messages are replayed.
  `excluded` messages are kept for display but never sent.
- Storage writes during a run use `st.db` (`context.WithoutCancel`) so Stop never
  leaves half-written state; LLM calls and handlers use the cancellable run ctx.
- A run is detached from the caller's ctx once the session is acquired
  (`run` wraps it in `context.WithoutCancel`): only Stop (or losing the lock)
  ends it.
- One run per session via the `agent_sessions.run_id` lock + heartbeat; a run that
  lost the lock (`ErrLockLost`) must not write anything.
- Crash safety relies on deterministic ids for derived rows (`msg_img_<call>`,
  `msg_<queueID>`) and on `heal()` at the start of every run.
- Calls of one turn run in parallel (`executeAll`, `Config.ToolConcurrency`); they
  only update pre-created rows, never insert messages, so ordering stays intact.
  On stop/timeout a handler's error result is replaced by the stop/timeout
  result (a ctx-aware handler errors *because* of the cancellation).
- `DeleteSession` keeps `agent_llm_calls` on purpose (billing, no content).
- A step reads only messages from the latest compaction point
  (`latestCompaction` + `messagesFromSeq`); never load a whole session.
- Every LLM request (chat and summary) goes through `beforeLLMCall` first.
- After a compaction, the last LLM call's token count no longer describes the
  window (`estimateTokens` checks this); usage rows of summary calls point at the
  compaction message, not an assistant message.

## Key files

- `client.go` — `NewClient` (creates tables, crash recovery) and all store-backed
  reads/ops: sessions, messages, queue, usage, billing, `Stop`.
- `config.go` — `Config` (client-wide), `AgentOptions`, `CompactionMode`, defaults.
- `agent.go` — `Agent` construction and public run methods (Chat, ChatMessage,
  Continue, Confirm, Enqueue).
- `run.go` — run lifecycle: acquire, heartbeat, heal, finish.
- `lock.go` — session lock (acquire/checkpoint/release).
- `stop.go` — non-blocking, idempotent Stop (marks `stopping`, cancels a
  local run) incl. taking over dead runs.
- `loop.go` — agent loop, call execution, tool message bookkeeping.
- `step.go` — one streamed LLM call, throttled flushes, usage recording.
- `compaction.go` — context window, token estimate, anchor/summary compaction.
- `queue.go` — queued user messages (claimed `taken` before insertion so
  `CancelQueued` never races a run), reminder injection.
- `rpc.go` / `prompt.go` / `jsonschema.go` — Method/Call/Typed/NewMethod, JSON-RPC
  dispatch and result limit, system prompt + tool definition, struct-tag schemas.
- `viewimage.go` — `view_image` tool and healing of unloadable image URLs.
- `log.go` — slog wiring (per-agent level filter), `LLMCallInfo`, the
  `BeforeLLMCall` hook and `callback` (every user callback goes through it:
  panics recovered); `OnToolCall` events (`OnRunEnd` fires in `run`).
  Info = lifecycle/errors, Debug = per request/call.
- `docs.go` — mounted docs (`Config.Docs` fs.FS, frontmatter parser, `read_doc`
  paging) and `SystemPromptFile`; loaded once per Client into `resources`.
- `llm.go` / `stream.go` — HTTP client with retries, SSE parsing/accumulation.
- `*_store.go`, `session.go`, `schema.go`, `recovery.go` — persistence, DDL,
  crash recovery (`resetSession`).
- `billing.go` / `settle.go` — usage parsing, pricing, settlement (claim/charge).
- `examples/` — separate module: runnable example (`basic/`) and deterministic
  end-to-end tests against a fake streaming OpenAI server + SQLite.
- `demo/` — **git-ignored**, live e2e tests against the real LLM proxy; never commit.

## Development

```bash
go vet ./... && go test -race ./...                  # root unit tests
cd examples && go test -race ./...                   # e2e with fake LLM + SQLite
cd demo && go test -v -timeout 40m .                 # live LLM, needs LLM_API_KEY
cd demo && go test -v -run 'TestStop|TestCrash' .    # subset
```

- Live tests default to `https://llm.iamdev.me/v1`, model `low-text` (supports tool
  calls and images); override with `LLM_URL` / `LLM_MODEL`. Live model behaviour is
  not deterministic: assert structure, not wording.
- When a live test finds an SDK bug, also add a deterministic regression test in
  `examples/` (the fake server can script replies, HTTP errors, in-stream errors,
  hangs and per-reply prompt token counts).
- Releases are git tags (`vX.Y.Z`); push with `git push origin master --tags`.
