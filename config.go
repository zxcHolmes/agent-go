package agent

import (
	"context"
	"io/fs"
	"log/slog"
	"net/http"
	"time"
)

// Config is the client-wide configuration shared by every agent created from
// a Client. Per-session values (session id, context params) are passed to
// Client.Agent; AgentOptions.Override can adjust a copy of this config for a
// single agent.
type Config struct {
	// Store persists sessions, messages, rpc calls and usage. Required.
	Store Store
	// Billing prices every LLM call in credits. Optional.
	Billing Billing

	// SkipSchema makes NewClient skip creating tables (use SchemaStatements
	// with your own migration tool instead).
	SkipSchema bool
	// RecoverStaleAfter limits the crash recovery done by NewClient to
	// sessions without a heartbeat for this long. 0 recovers every running
	// session, which is right for a single process. With several processes
	// sharing one database, set it above StaleAfter so a restarting instance
	// does not reset runs that are alive elsewhere.
	RecoverStaleAfter time.Duration

	// The fields below configure agents; BaseURL and Model are required only
	// when creating an agent.

	// OpenAI-compatible endpoint, e.g. "https://api.openai.com/v1".
	// "/chat/completions" is appended unless already present.
	BaseURL string
	APIKey  string
	Model   string

	// ContextLength is the model context window in tokens. When set, the agent
	// estimates the prompt size before each call, compacts the history when
	// it passes CompactionThreshold (see Compaction), caps max_tokens to the
	// remaining room, and fails with ErrContextLengthExceeded when even the
	// compacted history does not fit. 0 disables all of this.
	ContextLength int
	// MaxOutputTokens is sent as max_tokens (0 = provider default).
	MaxOutputTokens int
	// UseMaxCompletionTokens sends max_completion_tokens instead of max_tokens
	// (required by some newer OpenAI models).
	UseMaxCompletionTokens bool

	// SystemPrompt is prepended to every request. Keep it stable across calls
	// so the provider prefix cache keeps hitting.
	SystemPrompt string
	// SystemPromptFile loads the core system prompt from a markdown file
	// (frontmatter is stripped), so long prompts stay out of Go code. The path
	// is inside Docs when Docs is set, otherwise on the local filesystem.
	// SystemPrompt, if also set, is appended after it.
	SystemPromptFile string
	// Docs mounts a tree of markdown documents (e.g. os.DirFS("docs") or an
	// embed.FS). Each file needs frontmatter with a unique "title" and
	// usually a "description"; the frontmatter of every document is listed
	// in the system prompt and the model reads bodies on demand with the
	// read_doc tool. Loaded once by NewClient.
	Docs fs.FS
	// DocPageChars splits long documents into pages of this many characters
	// for read_doc (default 20000).
	DocPageChars int
	// RPCDoc is free-form documentation of your JSON-RPC API, appended to the
	// system prompt after the auto-generated method list.
	RPCDoc string
	// Methods are the RPC methods the model may call.
	Methods []Method

	// StreamFlushInterval is how often a streaming assistant message is
	// written to the store (default 2s). The message row is created on the
	// first delta and always written once more when the stream ends.
	StreamFlushInterval time.Duration
	// StreamIdleTimeout aborts a stream that sends nothing for this long
	// (default 2m, negative disables).
	StreamIdleTimeout time.Duration
	// KeepReasoning includes streamed reasoning text as "reasoning_content" in
	// the assistant message replayed to the model. Some providers require it
	// (e.g. DeepSeek thinking mode with tool calls), others reject it. The
	// reasoning is always stored in Message.Reasoning either way.
	KeepReasoning bool

	// Compaction selects how the history is shortened when a request would
	// use more than CompactionThreshold of ContextLength (default
	// CompactionAnchor). Requires ContextLength.
	Compaction CompactionMode
	// CompactionThreshold is the fraction of ContextLength that triggers
	// compaction before an LLM call (default 0.8).
	CompactionThreshold float64
	// CompactionPrompt is the instruction given to the model when writing a
	// summary in CompactionSummary mode (a sensible default is used).
	CompactionPrompt string

	// Reminder is appended to every user message as
	// "<reminder>...</reminder>" (Chat, ChatMessage and queued messages). It
	// is stored in the raw message sent to the model; Message.Content keeps
	// the text without it.
	Reminder string

	// RPCTimeout bounds how long one RPC call may run; Method.Timeout
	// overrides it per method. When it expires the handler's ctx is cancelled
	// and the model gets a CodeTimeout error right away (a handler that
	// ignores ctx keeps running in the background; its result is discarded).
	// 0 (default) means no timeout.
	RPCTimeout time.Duration

	// BeforeLLMCall is called before every LLM request, including summary
	// calls made by compaction. Returning an error cancels the request and
	// ends the run with that error (e.g. insufficient credits); nothing is
	// spent. See LLMCallInfo.
	BeforeLLMCall func(ctx context.Context, info LLMCallInfo) error

	// CacheControl adds Anthropic-style prompt caching breakpoints
	// ({"cache_control":{"type":"ephemeral"}}) to the system prompt and the
	// last message of every request, as OpenRouter and Anthropic-compatible
	// gateways expect for Claude models. Other providers cache prefixes
	// automatically and do not need it. Stored messages are not modified.
	CacheControl bool

	// Logger receives the SDK's logs (default slog.Default()). LogLevel sets
	// the minimum level for this agent: slog.LevelInfo (default) logs runs,
	// compactions, stops, recoveries and errors; slog.LevelDebug adds every
	// LLM request, tool call, queue drain and retry.
	Logger   *slog.Logger
	LogLevel slog.Level

	// StopPollInterval is how often a running agent checks the store for a
	// Stop issued by another process (default 1s). Stop in the same process
	// is immediate regardless. Negative disables polling: cross-process
	// stops are then noticed before the next step only.
	StopPollInterval time.Duration

	// MaxRPCResultChars caps the size (in characters) of what one RPC call
	// returns to the model; longer results and error messages are cut with a
	// note saying how much was dropped. 0 means no limit.
	MaxRPCResultChars int

	// ToolName is the name of the JSON-RPC tool (default "json_rpc").
	ToolName string
	// ViewImage enables the built-in view_image tool (see ViewImageTool): the
	// model passes an image URL and gets to see the image. Only enable it for
	// vision-capable models; others reject image_url content.
	ViewImage bool
	// ViewImageDetail is sent as image_url.detail ("low", "high" or "auto");
	// empty omits it. "low" is much cheaper in image tokens.
	ViewImageDetail string
	// MaxSteps bounds LLM calls per run (default 50).
	MaxSteps int
	// ExtraBody is merged into every request body (temperature, top_p,
	// reasoning_effort, ...). It cannot override model, messages, tools or
	// stream; a nil value removes a default field such as "stream_options".
	ExtraBody map[string]any
	// Headers are added to every LLM request.
	Headers    map[string]string
	HTTPClient *http.Client
	// MaxRetries for 429/5xx/network errors before any output was streamed
	// (default 2, negative disables).
	MaxRetries int
	// StaleAfter is how long a "running" session may go without a heartbeat
	// before another run may take it over (default 1m). Runs heartbeat in the
	// background every few seconds.
	StaleAfter time.Duration

	// OnStream is called for every streamed text delta (the store is only
	// written every StreamFlushInterval).
	OnStream func(ctx context.Context, messageID, content, reasoning string)
	// OnMessage is called after a message is inserted or updated in the store,
	// including throttled streaming flushes and tool status changes.
	OnMessage func(ctx context.Context, m Message)
	// OnUsage is called after each LLM call is recorded, e.g. to deduct credits.
	OnUsage func(ctx context.Context, r UsageRecord)
}

// AgentOptions are the per-agent inputs of Client.Agent.
type AgentOptions struct {
	// ContextParams are passed to every RPC handler through
	// Call.ContextParams (e.g. the authenticated user, tenant, token). They
	// are not sent to the model and not persisted; pass the current request's
	// values every time you build an agent.
	ContextParams map[string]any
	// Override adjusts a copy of the client config for this agent only, e.g.
	// a different system prompt, model or method set. Changing Store has no
	// effect.
	Override func(cfg *Config)
}

// withDefaults fills in zero values.
// CompactionMode selects the context compaction strategy.
type CompactionMode string

const (
	// CompactionAnchor (default) moves the start of the history sent to the
	// model to the most recent user message. Older messages stay stored and
	// visible but are no longer sent. The start only moves on compaction, so
	// the provider prefix cache keeps working between compactions.
	CompactionAnchor CompactionMode = "anchor"
	// CompactionSummary does the same, but first asks the model to summarize
	// the dropped messages; the summary is sent in their place.
	CompactionSummary CompactionMode = "summary"
	// CompactionOff never compacts; requests that no longer fit fail with
	// ErrContextLengthExceeded.
	CompactionOff CompactionMode = "off"
)

func (cfg Config) withDefaults() Config {
	if cfg.StopPollInterval == 0 {
		cfg.StopPollInterval = time.Second
	}
	if cfg.DocPageChars <= 0 {
		cfg.DocPageChars = 20000
	}
	if cfg.Compaction == "" {
		cfg.Compaction = CompactionAnchor
	}
	if cfg.CompactionThreshold <= 0 || cfg.CompactionThreshold >= 1 {
		cfg.CompactionThreshold = 0.8
	}
	if cfg.ToolName == "" {
		cfg.ToolName = "json_rpc"
	}
	if cfg.MaxSteps <= 0 {
		cfg.MaxSteps = 50
	}
	if cfg.MaxRetries == 0 {
		cfg.MaxRetries = 2
	} else if cfg.MaxRetries < 0 {
		cfg.MaxRetries = 0
	}
	if cfg.StaleAfter <= 0 {
		cfg.StaleAfter = time.Minute
	}
	if cfg.StreamFlushInterval <= 0 {
		cfg.StreamFlushInterval = 2 * time.Second
	}
	if cfg.StreamIdleTimeout == 0 {
		cfg.StreamIdleTimeout = 2 * time.Minute
	} else if cfg.StreamIdleTimeout < 0 {
		cfg.StreamIdleTimeout = 0
	}
	return cfg
}
