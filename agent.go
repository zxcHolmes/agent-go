package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
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
	// estimates the prompt size before each call, fails with
	// ErrContextLengthExceeded when it no longer fits, and caps max_tokens to
	// the remaining room. 0 disables the check.
	ContextLength int
	// MaxOutputTokens is sent as max_tokens (0 = provider default).
	MaxOutputTokens int
	// UseMaxCompletionTokens sends max_completion_tokens instead of max_tokens
	// (required by some newer OpenAI models).
	UseMaxCompletionTokens bool

	// SystemPrompt is prepended to every request. Keep it stable across calls
	// so the provider prefix cache keeps hitting.
	SystemPrompt string
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

	// ToolName is the name of the single JSON-RPC tool (default "json_rpc").
	ToolName string
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

// Agent runs the tool-calling loop for one session. Create it with
// Client.Agent. It is safe for concurrent use; at most one run per session
// executes at a time (across processes too, through the session row lock).
type Agent struct {
	client        *Client
	cfg           Config
	store         Store
	llm           *llmClient
	methods       map[string]Method
	system        json.RawMessage
	tools         []json.RawMessage
	sessionID     string
	contextParams map[string]any
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
func (cfg Config) withDefaults() Config {
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

func newAgent(ctx context.Context, c *Client, sessionID string, opts AgentOptions) (*Agent, error) {
	cfg := c.cfg
	cfg.Methods = append([]Method(nil), c.cfg.Methods...)
	if opts.Override != nil {
		opts.Override(&cfg)
	}
	cfg.Store = c.store
	cfg = cfg.withDefaults()
	if cfg.BaseURL == "" || cfg.Model == "" {
		return nil, errors.New("agent: Config.BaseURL and Config.Model are required to create an agent")
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{} // no overall timeout: streams can be long; see StreamIdleTimeout
	}
	a := &Agent{
		client:        c,
		cfg:           cfg,
		store:         c.store,
		methods:       make(map[string]Method, len(cfg.Methods)),
		contextParams: opts.ContextParams,
		llm: &llmClient{
			endpoint:    chatEndpoint(cfg.BaseURL),
			apiKey:      cfg.APIKey,
			headers:     cfg.Headers,
			http:        httpClient,
			maxRetries:  cfg.MaxRetries,
			idleTimeout: cfg.StreamIdleTimeout,
		},
	}
	for _, m := range cfg.Methods {
		if m.Name == "" || m.Handler == nil {
			return nil, fmt.Errorf("agent: method %q needs a name and a handler", m.Name)
		}
		if _, dup := a.methods[m.Name]; dup {
			return nil, fmt.Errorf("agent: duplicate method %q", m.Name)
		}
		a.methods[m.Name] = m
	}

	prompt, err := a.buildSystemPrompt()
	if err != nil {
		return nil, err
	}
	if prompt != "" {
		a.system, _ = marshalJSON(struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		}{"system", prompt})
	}
	if len(a.methods) > 0 {
		tool, err := a.buildTool()
		if err != nil {
			return nil, err
		}
		a.tools = []json.RawMessage{tool}
	}

	if sessionID == "" {
		if a.sessionID, err = createSession(ctx, a.store, nil); err != nil {
			return nil, err
		}
	} else {
		if _, err := getSession(ctx, a.store, sessionID); err != nil {
			return nil, err
		}
		a.sessionID = sessionID
	}
	return a, nil
}

// Client returns the client this agent was created from.
func (a *Agent) Client() *Client { return a.client }

// SessionID returns the id of the session this agent drives.
func (a *Agent) SessionID() string { return a.sessionID }

// Session loads the session row.
func (a *Agent) Session(ctx context.Context) (*Session, error) {
	return getSession(ctx, a.store, a.sessionID)
}

// Status returns the current session status.
func (a *Agent) Status(ctx context.Context) (Status, error) {
	s, err := a.Session(ctx)
	if err != nil {
		return "", err
	}
	return s.Status, nil
}

// PendingCalls returns the RPC calls waiting for confirmation.
func (a *Agent) PendingCalls(ctx context.Context) ([]RPCCall, error) {
	return pendingCalls(ctx, a.store, a.sessionID)
}

// Chat appends a user message and runs the agent loop until the model answers
// without tool calls, a call needs confirmation, Stop is called, or MaxSteps
// is reached. It blocks for the whole run; poll the store for progress.
func (a *Agent) Chat(ctx context.Context, prompt string) (*RunResult, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("agent: empty prompt")
	}
	raw, err := marshalJSON(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{"user", prompt})
	if err != nil {
		return nil, err
	}
	return a.ChatMessage(ctx, raw)
}

// ChatMessage is Chat with a caller-built user message, e.g. multimodal
// content: {"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url",...}]}.
func (a *Agent) ChatMessage(ctx context.Context, raw json.RawMessage) (*RunResult, error) {
	if !json.Valid(raw) {
		return nil, errors.New("agent: user message is not valid JSON")
	}
	return a.run(ctx, []Status{StatusIdle}, func(ctx context.Context, st *runState) error {
		open, err := openCalls(st.db, a.store, a.sessionID)
		if err != nil {
			return err
		}
		for _, c := range open {
			if c.Status == CallAwaitingConfirmation {
				return ErrWaitingConfirmation
			}
		}
		// Leftovers from an interrupted run: close them so the history stays valid.
		if err := a.cancelCalls(ctx, st, open, false, &RPCError{Code: CodeCancelled, Message: "cancelled: superseded by a new user message"}); err != nil {
			return err
		}
		m := newMessage(a.sessionID, raw, MessageDone)
		return a.insertMessage(ctx, st, &m)
	}, true)
}

// Continue resumes the loop without a new user message, e.g. after Stop, an
// error, a crash or a truncated answer.
func (a *Agent) Continue(ctx context.Context) (*RunResult, error) {
	return a.run(ctx, []Status{StatusIdle}, nil, true)
}

// Confirm approves or rejects calls awaiting confirmation. Once no call is
// awaiting anymore, the loop continues and Confirm blocks like Chat.
func (a *Agent) Confirm(ctx context.Context, decisions ...Decision) (*RunResult, error) {
	if len(decisions) == 0 {
		return nil, errors.New("agent: no decisions")
	}
	return a.run(ctx, []Status{StatusWaitingConfirmation}, func(ctx context.Context, st *runState) error {
		pending, err := pendingCalls(st.db, a.store, a.sessionID)
		if err != nil {
			return err
		}
		byID := make(map[string]*RPCCall, len(pending))
		for i := range pending {
			byID[pending[i].ID] = &pending[i]
		}
		for _, d := range decisions {
			if byID[d.CallID] == nil {
				return fmt.Errorf("%w: %s", ErrCallNotPending, d.CallID)
			}
		}
		for _, d := range decisions {
			c := byID[d.CallID]
			if d.Approve {
				err = setCallStatus(st.db, a.store, c, CallApproved)
			} else {
				msg := "rejected by user"
				if d.Reason != "" {
					msg += ": " + d.Reason
				}
				err = a.completeCall(ctx, st, c, CallRejected, rpcErrorResponse(c.RPCID, &RPCError{Code: CodeRejected, Message: msg}))
			}
			if err != nil {
				return err
			}
		}
		return nil
	}, true)
}

// Stop stops the session and blocks until it is idle. It is idempotent and
// safe to call concurrently, from any process sharing the store.
//
//   - running: the session turns "stopping"; the run is interrupted at once. A
//     streaming answer is cut off and keeps its partial text ("interrupted"),
//     the LLM request is cancelled, a running RPC call is abandoned without
//     waiting for its handler (the handler's ctx is cancelled and its eventual
//     result discarded), and every unfinished call, including calls awaiting
//     confirmation, gets a "stopped by user" error result.
//   - waiting_confirmation: pending calls get "stopped by user".
//   - stopping: waits for the stop in progress.
//   - idle: returns nil immediately.
//
// A run whose process died (no heartbeat for Config.StaleAfter) is taken over
// and cleaned up by Stop itself. If ctx ends first, Stop returns its error;
// the session still finishes stopping on its own.
func (a *Agent) Stop(ctx context.Context) error {
	for {
		s, err := getSession(ctx, a.store, a.sessionID)
		if err != nil {
			return err
		}
		switch s.Status {
		case StatusIdle:
			return nil
		case StatusWaitingConfirmation:
			if err := a.stopIdle(ctx); err != nil && !errors.Is(err, ErrBusy) && !errors.Is(err, ErrNoPendingCalls) {
				return err
			}
			continue // re-read: someone may have raced us
		case StatusRunning, StatusStopping:
			if s.Status == StatusRunning {
				// updated_at is left alone so a dead run still looks stale.
				if _, err := a.store.Exec(ctx, "UPDATE agent_sessions SET status = ? WHERE id = ? AND status = ?",
					string(StatusStopping), a.sessionID, string(StatusRunning)); err != nil {
					return err
				}
			}
			if v, ok := running.Load(a.sessionID); ok {
				v.(*localRun).cancel(ErrStopped)
			}
			if time.Since(s.UpdatedAt) > a.cfg.StaleAfter {
				// The owning process is gone: take over and clean up here.
				if err := a.stopIdle(ctx); err != nil && !errors.Is(err, ErrBusy) && !errors.Is(err, ErrNoPendingCalls) {
					return err
				}
				continue
			}
		}
		t := time.NewTimer(20 * time.Millisecond)
		select {
		case <-ctx.Done():
			t.Stop()
			return context.Cause(ctx)
		case <-t.C:
		}
	}
}

// stopIdle takes the session (waiting for confirmation, or abandoned by a dead
// run), repairs it, cancels every unfinished call and leaves it idle.
func (a *Agent) stopIdle(ctx context.Context) error {
	_, err := a.run(ctx, []Status{StatusWaitingConfirmation}, nil, false)
	return err
}

// ---- run machinery ----

var running sync.Map // session id -> *localRun

type localRun struct {
	runID  string
	cancel context.CancelCauseFunc
}

type runState struct {
	runID    string
	db       context.Context // not cancelled by Stop, so bookkeeping always completes
	cancel   context.CancelCauseFunc
	startSeq int64 // messages with a greater seq were produced by this run
	res      *RunResult
}

// run acquires the session and drives it. With loop=false it only heals the
// session and stops it (used by Stop).
func (a *Agent) run(ctx context.Context, from []Status, prepare func(context.Context, *runState) error, loop bool) (*RunResult, error) {
	runID, err := a.acquire(ctx, from)
	if err != nil {
		return nil, err
	}
	runCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	lr := &localRun{runID: runID, cancel: cancel}
	running.Store(a.sessionID, lr)
	defer running.CompareAndDelete(a.sessionID, lr)

	st := &runState{runID: runID, db: context.WithoutCancel(ctx), cancel: cancel, res: &RunResult{SessionID: a.sessionID}}
	hbDone := a.heartbeat(runCtx, st)
	defer hbDone()

	var reason StopReason
	err = a.heal(runCtx, st)
	if err == nil && prepare != nil {
		err = prepare(runCtx, st)
	}
	switch {
	case err != nil:
	case loop:
		reason, err = a.loop(runCtx, st)
	default:
		st.cancel(ErrStopped)
		err = ErrStopped
	}
	hbDone()
	return a.finish(runCtx, st, reason, err)
}

// heartbeat keeps the session lock fresh while the run is alive and watches
// for Stop from other processes (status "stopping"), so a stop interrupts even
// the middle of a stream or a long RPC call within ~500ms. The returned func
// stops it and waits for it to exit.
func (a *Agent) heartbeat(ctx context.Context, st *runState) func() {
	beatEvery := a.cfg.StaleAfter / 4
	if beatEvery > 5*time.Second {
		beatEvery = 5 * time.Second
	}
	pollEvery := 500 * time.Millisecond
	if beatEvery < pollEvery {
		pollEvery = beatEvery
	}
	if pollEvery < 50*time.Millisecond {
		pollEvery = 50 * time.Millisecond
	}
	quit := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		t := time.NewTicker(pollEvery)
		defer t.Stop()
		lastBeat := time.Now()
		for {
			select {
			case <-quit:
				return
			case <-ctx.Done():
				return
			case <-t.C:
			}
			if time.Since(lastBeat) >= beatEvery {
				lastBeat = time.Now()
				_, _ = a.store.Exec(st.db, "UPDATE agent_sessions SET updated_at = ? WHERE id = ? AND run_id = ?", nowMillis(), a.sessionID, st.runID)
			}
			owner, status, err := a.sessionLock(st.db)
			switch {
			case err != nil:
			case owner != st.runID:
				st.cancel(ErrLockLost)
				return
			case status == StatusStopping:
				st.cancel(ErrStopped)
				return
			}
		}
	}()
	var once sync.Once
	return func() {
		once.Do(func() { close(quit) })
		<-exited
	}
}

// heal brings the session back to a consistent state before running: it
// repairs leftovers of a crashed run (for sessions taken over without Init)
// and records tool calls whose records or tool messages were never written.
func (a *Agent) heal(ctx context.Context, st *runState) error {
	if err := recoverSessionData(st.db, a.store, a.sessionID); err != nil {
		return err
	}
	last, err := latestMessages(st.db, a.store, a.sessionID, 1)
	if err != nil {
		return err
	}
	if len(last) > 0 {
		st.startSeq = last[0].Seq
	}
	return a.reconcileCalls(ctx, st)
}

// reconcileCalls makes sure every tool call of the latest assistant message
// has an RPC call record and a tool message (a crash between writes can leave
// either missing).
func (a *Agent) reconcileCalls(ctx context.Context, st *runState) error {
	msg, err := lastAssistant(st.db, a.store, a.sessionID)
	if err != nil || msg == nil || len(msg.ToolCalls) == 0 {
		return err
	}
	calls, err := callsForMessage(st.db, a.store, msg.ID)
	if err != nil {
		return err
	}
	byIndex := make(map[int]*RPCCall, len(calls))
	for i := range calls {
		byIndex[calls[i].Index] = &calls[i]
	}
	for i, tc := range msg.ToolCalls {
		c := byIndex[i]
		if c == nil {
			nc := a.newCall(msg, i, tc)
			if err := a.createCall(ctx, st, &nc); err != nil {
				return err
			}
			continue
		}
		if _, err := getMessage(st.db, a.store, a.sessionID, c.ResultMessageID); errors.Is(err, ErrMessageNotFound) {
			m := a.toolMessage(c)
			m.ID = c.ResultMessageID
			if err := a.insertMessage(ctx, st, &m); err != nil {
				return err
			}
		} else if err != nil {
			return err
		}
	}
	return nil
}

func (a *Agent) finish(runCtx context.Context, st *runState, reason StopReason, err error) (*RunResult, error) {
	res := st.res
	if errors.Is(err, ErrLockLost) || errors.Is(context.Cause(runCtx), ErrLockLost) {
		return res, ErrLockLost
	}
	interrupted := err != nil && runCtx.Err() != nil
	stopped := interrupted && errors.Is(context.Cause(runCtx), ErrStopped)
	if !stopped {
		// Stop may have landed just as the run ended on its own; honour it.
		if owner, status, lerr := a.sessionLock(st.db); lerr == nil && owner == st.runID && status == StatusStopping {
			stopped = true
		}
	}
	if stopped {
		// A requested stop is a normal outcome, not an error.
		err = nil
	}
	var bookErr error // bookkeeping failures while wrapping up
	if stopped || interrupted {
		open, oerr := openCalls(st.db, a.store, a.sessionID)
		if oerr == nil {
			if stopped {
				oerr = a.cancelCalls(runCtx, st, open, true, errStoppedNotRun)
			} else {
				oerr = a.cancelCalls(runCtx, st, open, false, &RPCError{Code: CodeCancelled, Message: "cancelled before this call ran: " + context.Cause(runCtx).Error()})
			}
		}
		bookErr = errors.Join(bookErr, oerr)
	}
	pending, perr := pendingCalls(st.db, a.store, a.sessionID)
	bookErr = errors.Join(bookErr, perr)
	status := StatusIdle
	if len(pending) > 0 {
		status = StatusWaitingConfirmation
	}
	lastErr := ""
	if err != nil && !errors.Is(err, ErrWaitingConfirmation) {
		lastErr = err.Error()
	}
	bookErr = errors.Join(bookErr, a.release(st, status, lastErr))
	msgs, merr := messagesAfterSeq(st.db, a.store, a.sessionID, st.startSeq)
	bookErr = errors.Join(bookErr, merr)
	if bookErr != nil {
		err = errors.Join(err, bookErr)
	}
	res.Status, res.PendingCalls, res.Messages = status, pending, msgs
	switch {
	case err != nil:
	case stopped:
		res.StopReason = StopStopped
	case status == StatusWaitingConfirmation:
		res.StopReason = StopWaitingConfirmation
	default:
		res.StopReason = reason
	}
	return res, err
}

var (
	errStoppedNotRun  = &RPCError{Code: CodeCancelled, Message: "stopped by user: this call was not executed"}
	errStoppedRunning = &RPCError{Code: CodeCancelled, Message: "stopped by user while this call was running; it may or may not have taken effect"}
)

// acquire atomically moves the session to running if it is in one of the
// allowed states, or if a previous run stopped heartbeating (crash).
func (a *Agent) acquire(ctx context.Context, from []Status) (string, error) {
	runID := newID("run")
	now := nowMillis()
	args := []any{string(StatusRunning), runID, now, a.sessionID}
	for _, s := range from {
		args = append(args, string(s))
	}
	args = append(args, string(StatusRunning), string(StatusStopping), now-a.cfg.StaleAfter.Milliseconds())
	ph := strings.TrimSuffix(strings.Repeat("?, ", len(from)), ", ")
	if _, err := a.store.Exec(ctx,
		"UPDATE agent_sessions SET status = ?, run_id = ?, last_error = '', updated_at = ? WHERE id = ? AND (status IN ("+ph+") OR (status IN (?, ?) AND updated_at < ?))",
		args...); err != nil {
		return "", err
	}
	// Verify by reading back: rows-affected is unreliable across drivers.
	owner, _, err := a.sessionLock(ctx)
	if err != nil {
		return "", err
	}
	if owner == runID {
		return runID, nil
	}
	s, err := getSession(ctx, a.store, a.sessionID)
	if err != nil {
		return "", err
	}
	switch s.Status {
	case StatusRunning, StatusStopping:
		return "", ErrBusy
	case StatusWaitingConfirmation:
		return "", ErrWaitingConfirmation
	default:
		return "", ErrNoPendingCalls
	}
}

func (a *Agent) sessionLock(ctx context.Context) (runID string, status Status, err error) {
	rows, err := a.store.Query(ctx, "SELECT run_id, status FROM agent_sessions WHERE id = ?", a.sessionID)
	if err != nil {
		return "", "", err
	}
	defer rows.Close()
	if !rows.Next() {
		if err := rows.Err(); err != nil {
			return "", "", err
		}
		return "", "", ErrSessionNotFound
	}
	var st string
	err = rows.Scan(&runID, &st)
	return runID, Status(st), err
}

// checkpoint heartbeats the lock and honours stop requests.
func (a *Agent) checkpoint(ctx context.Context, st *runState) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	if _, err := a.store.Exec(st.db, "UPDATE agent_sessions SET updated_at = ? WHERE id = ? AND run_id = ?", nowMillis(), a.sessionID, st.runID); err != nil {
		return err
	}
	owner, status, err := a.sessionLock(st.db)
	if err != nil {
		return err
	}
	if owner != st.runID {
		return ErrLockLost
	}
	if status == StatusStopping {
		st.cancel(ErrStopped)
		return ErrStopped
	}
	return nil
}

// ownsLock reports whether this run still holds the session, e.g. after a
// long RPC call during which Init may have recovered the session.
func (a *Agent) ownsLock(st *runState) error {
	owner, _, err := a.sessionLock(st.db)
	if err != nil {
		return err
	}
	if owner != st.runID {
		return ErrLockLost
	}
	return nil
}

func (a *Agent) release(st *runState, status Status, lastErr string) error {
	_, err := a.store.Exec(st.db, "UPDATE agent_sessions SET status = ?, run_id = '', last_error = ?, updated_at = ? WHERE id = ? AND run_id = ?",
		string(status), lastErr, nowMillis(), a.sessionID, st.runID)
	return err
}

func (a *Agent) loop(ctx context.Context, st *runState) (StopReason, error) {
	llmCalls := 0
	for {
		if err := a.checkpoint(ctx, st); err != nil {
			return "", err
		}
		open, err := openCalls(st.db, a.store, a.sessionID)
		if err != nil {
			return "", err
		}
		if len(open) > 0 {
			waiting := false
			for i := range open {
				c := &open[i]
				switch c.Status {
				case CallQueued, CallApproved:
					if err := a.execute(ctx, st, c); err != nil {
						return "", err
					}
				case CallAwaitingConfirmation:
					waiting = true
				}
			}
			if waiting {
				return StopWaitingConfirmation, nil
			}
			continue
		}
		if llmCalls >= a.cfg.MaxSteps {
			return StopMaxSteps, nil
		}
		llmCalls++
		hasCalls, err := a.step(ctx, st)
		if err != nil {
			return "", err
		}
		if !hasCalls {
			return StopCompleted, nil
		}
	}
}

// execute runs one call. The tool message is marked "running" before the
// handler starts, so a crash mid-call is visible and recoverable. If the run
// is stopped meanwhile, the call is closed at once without waiting for the
// handler, whose late result is discarded.
func (a *Agent) execute(ctx context.Context, st *runState, c *RPCCall) error {
	if err := a.checkpoint(ctx, st); err != nil {
		return err
	}
	if err := setCallStatus(st.db, a.store, c, CallRunning); err != nil {
		return err
	}
	a.notify(ctx, st, c.ResultMessageID)
	done := make(chan json.RawMessage, 1)
	go func() { done <- a.invoke(ctx, c) }()
	status := CallDone
	var result json.RawMessage
	select {
	case result = <-done:
	case <-ctx.Done():
		select {
		case result = <-done: // finished at the same instant: keep the real result
		default:
			status = CallCancelled
			rerr := errStoppedRunning
			if !errors.Is(context.Cause(ctx), ErrStopped) {
				rerr = &RPCError{Code: CodeCancelled, Message: "cancelled while this call was running (" + context.Cause(ctx).Error() + "); it may or may not have taken effect"}
			}
			result = rpcErrorResponse(c.RPCID, rerr)
		}
	}
	if err := a.ownsLock(st); err != nil {
		return err // the session was recovered meanwhile; do not overwrite it
	}
	if err := a.completeCall(ctx, st, c, status, result); err != nil {
		return err
	}
	if status == CallCancelled {
		return context.Cause(ctx)
	}
	return nil
}

func (a *Agent) newCall(m *Message, i int, tc ToolCall) RPCCall {
	now := time.Now()
	c := RPCCall{
		ID: newID("call"), SessionID: a.sessionID, MessageID: m.ID, ToolCallID: tc.ID, Index: i,
		Method: tc.Function.Name, Status: CallQueued, ResultMessageID: newID("msg"), CreatedAt: now, UpdatedAt: now,
	}
	fallbackID, _ := marshalJSON(tc.ID)
	if tc.Function.Name != a.cfg.ToolName {
		c.Status = CallDone
		c.Result = rpcErrorResponse(nil, &RPCError{Code: CodeMethodNotFound, Message: fmt.Sprintf("unknown tool %q; the only tool is %q", tc.Function.Name, a.cfg.ToolName)})
		return c
	}
	req, rerr := parseRPCRequest(tc.Function.Arguments)
	c.Method, c.Params, c.RPCID = req.Method, req.Params, req.ID
	if len(c.RPCID) == 0 || string(c.RPCID) == "null" {
		c.RPCID = fallbackID
	}
	if rerr != nil {
		c.Status, c.Result = CallDone, rpcErrorResponse(c.RPCID, rerr)
		return c
	}
	m2, ok := a.methods[c.Method]
	if !ok {
		c.Status, c.Result = CallDone, rpcErrorResponse(c.RPCID, a.methodNotFound(c.Method))
		return c
	}
	if m2.RequireConfirm {
		c.RequireConfirm, c.Status = true, CallAwaitingConfirmation
	}
	return c
}

// toolMessage builds the tool message mirroring a call's current state.
func (a *Agent) toolMessage(c *RPCCall) Message {
	status, content := MessagePending, ""
	if c.Status == CallRunning {
		status = MessageRunning
	}
	if c.Status.finished() {
		status, content = MessageDone, string(c.Result)
	}
	return newMessage(a.sessionID, toolMessageRaw(c.ToolCallID, content), status)
}

// createCall writes a call record and its tool message.
func (a *Agent) createCall(ctx context.Context, st *runState, c *RPCCall) error {
	if err := insertCall(st.db, a.store, c); err != nil {
		return err
	}
	m := a.toolMessage(c)
	m.ID = c.ResultMessageID
	return a.insertMessage(ctx, st, &m)
}

func (a *Agent) completeCall(ctx context.Context, st *runState, c *RPCCall, status CallStatus, result json.RawMessage) error {
	if err := completeCall(st.db, a.store, c, status, result); err != nil {
		return err
	}
	a.notify(ctx, st, c.ResultMessageID)
	return nil
}

// cancelCalls closes calls that never started with rerr. Calls awaiting
// confirmation are closed too when includeAwaiting is set (Stop).
func (a *Agent) cancelCalls(ctx context.Context, st *runState, open []RPCCall, includeAwaiting bool, rerr *RPCError) error {
	for i := range open {
		c := &open[i]
		switch c.Status {
		case CallQueued, CallApproved:
		case CallAwaitingConfirmation:
			if !includeAwaiting {
				continue
			}
		default:
			continue
		}
		if err := a.completeCall(ctx, st, c, CallCancelled, rpcErrorResponse(c.RPCID, rerr)); err != nil {
			return err
		}
	}
	return nil
}

func (a *Agent) insertMessage(ctx context.Context, st *runState, m *Message) error {
	if err := insertMessage(st.db, a.store, m); err != nil {
		return err
	}
	if a.cfg.OnMessage != nil {
		a.cfg.OnMessage(ctx, *m)
	}
	return nil
}

func (a *Agent) updateMessage(ctx context.Context, st *runState, m *Message) error {
	if err := updateMessage(st.db, a.store, m); err != nil {
		return err
	}
	if a.cfg.OnMessage != nil {
		a.cfg.OnMessage(ctx, *m)
	}
	return nil
}

// notify reloads a message changed in the store and passes it to OnMessage.
func (a *Agent) notify(ctx context.Context, st *runState, messageID string) {
	if a.cfg.OnMessage == nil {
		return
	}
	if m, err := getMessage(st.db, a.store, a.sessionID, messageID); err == nil {
		a.cfg.OnMessage(ctx, *m)
	}
}

// assistantRaw builds the assistant message stored and replayed to the model.
func (a *Agent) assistantRaw(content, reasoning string, calls []ToolCall) json.RawMessage {
	msg := struct {
		Role             string     `json:"role"`
		Content          *string    `json:"content"`
		ReasoningContent string     `json:"reasoning_content,omitempty"`
		ToolCalls        []ToolCall `json:"tool_calls,omitempty"`
	}{Role: "assistant", ToolCalls: calls}
	if content != "" || len(calls) == 0 {
		msg.Content = &content
	}
	if a.cfg.KeepReasoning {
		msg.ReasoningContent = reasoning
	}
	for i := range msg.ToolCalls {
		if msg.ToolCalls[i].Type == "" {
			msg.ToolCalls[i].Type = "function"
		}
		if strings.TrimSpace(msg.ToolCalls[i].Function.Arguments) == "" {
			msg.ToolCalls[i].Function.Arguments = "{}"
		}
	}
	raw, _ := marshalJSON(msg)
	return raw
}

// step performs one streamed LLM call. The assistant message is created on
// the first delta, flushed every StreamFlushInterval, and finalised when the
// stream ends (or marked interrupted if it breaks).
func (a *Agent) step(ctx context.Context, st *runState) (bool, error) {
	history, err := allMessages(st.db, a.store, a.sessionID)
	if err != nil {
		return false, err
	}
	history = replayable(history)
	maxTokens, err := a.maxTokens(st, history)
	if err != nil {
		return false, err
	}
	msgs := make([]json.RawMessage, 0, len(history)+1)
	if a.system != nil {
		msgs = append(msgs, a.system)
	}
	for _, m := range history {
		msgs = append(msgs, m.Raw)
	}
	req := chatRequest{Model: a.cfg.Model, Messages: msgs, Tools: a.tools}
	if a.cfg.UseMaxCompletionTokens {
		req.MaxCompletionTokens = maxTokens
	} else {
		req.MaxTokens = maxTokens
	}

	// Deltas arrive on this goroutine; a trailing timer flushes buffered text
	// when the stream stalls, so the store never lags more than one interval.
	var (
		mu                 sync.Mutex
		msg                *Message
		content, reasoning strings.Builder
		lastFlush          time.Time
		timer              *time.Timer
		dirty, ended       bool
		flushErr           error
	)
	flush := func(status MessageStatus, raw json.RawMessage) error { // mu held
		lastFlush, dirty = time.Now(), false
		if msg == nil {
			m := newMessage(a.sessionID, raw, status)
			m.Reasoning = reasoning.String()
			msg = &m
			return a.insertMessage(ctx, st, msg)
		}
		msg.Raw, msg.Status, msg.Reasoning = raw, status, reasoning.String()
		return a.updateMessage(ctx, st, msg)
	}
	partial := func() json.RawMessage { return a.assistantRaw(content.String(), "", nil) }
	onDelta := func(dc, dr string) error {
		mu.Lock()
		defer mu.Unlock()
		if flushErr != nil {
			return flushErr
		}
		content.WriteString(dc)
		reasoning.WriteString(dr)
		dirty = true
		if wait := a.cfg.StreamFlushInterval - time.Since(lastFlush); msg == nil || wait <= 0 {
			if err := flush(MessageStreaming, partial()); err != nil {
				return err
			}
		} else if timer == nil {
			timer = time.AfterFunc(wait, func() {
				mu.Lock()
				defer mu.Unlock()
				timer = nil
				if dirty && !ended {
					flushErr = flush(MessageStreaming, partial())
				}
			})
		}
		if a.cfg.OnStream != nil {
			a.cfg.OnStream(ctx, msg.ID, dc, dr)
		}
		return nil
	}
	endStream := func() { // stop the trailing timer; afterwards only this goroutine touches state
		mu.Lock()
		ended = true
		if timer != nil {
			timer.Stop()
		}
		mu.Unlock()
	}

	start := time.Now()
	res, err := a.llm.stream(ctx, req, a.cfg.ExtraBody, onDelta)
	endStream()
	mu.Lock() // pairs with a timer callback that may have just run
	defer mu.Unlock()
	if err != nil {
		if msg != nil {
			// Keep what was generated; partial tool calls are dropped.
			if ferr := flush(MessageInterrupted, partial()); ferr != nil {
				err = errors.Join(err, ferr)
			}
		}
		return false, err
	}
	latency := time.Since(start).Milliseconds()
	if err := a.ownsLock(st); err != nil {
		return false, err // recovered while streaming; leave the message interrupted
	}
	content.Reset()
	content.WriteString(res.Content)
	reasoning.Reset()
	reasoning.WriteString(res.Reasoning)
	if err := flush(MessageDone, a.assistantRaw(res.Content, res.Reasoning, res.ToolCalls)); err != nil {
		return false, err
	}

	usage := parseUsage(res.Usage)
	var cost Cost
	if a.cfg.Billing != nil {
		cost = a.cfg.Billing.Cost(a.cfg.Model, usage)
	}
	rec := UsageRecord{
		ID: newID("llm"), SessionID: a.sessionID, MessageID: msg.ID, Model: a.cfg.Model,
		ResponseID: res.ID, FinishReason: res.FinishReason, Usage: usage, RawUsage: res.Usage,
		Cost: cost, LatencyMs: latency, CreatedAt: time.Now(),
	}
	if err := insertUsage(st.db, a.store, &rec); err != nil {
		return false, err
	}
	st.res.Usage.add(usage)
	st.res.Cost.add(cost)
	st.res.FinishReason = res.FinishReason
	if a.cfg.OnUsage != nil {
		a.cfg.OnUsage(ctx, rec)
	}
	for i, tc := range msg.ToolCalls {
		c := a.newCall(msg, i, tc)
		if err := a.createCall(ctx, st, &c); err != nil {
			return false, err
		}
	}
	return len(msg.ToolCalls) > 0, nil
}

// replayable drops messages that must not be sent to the model: interrupted
// assistant messages with no text, and anything not final.
func replayable(ms []Message) []Message {
	out := ms[:0:0]
	for _, m := range ms {
		switch {
		case m.Status == MessageInterrupted && m.Content == "":
		case !m.Status.Final():
		default:
			out = append(out, m)
		}
	}
	return out
}

// maxTokens estimates the prompt size (exact token count of the previous
// call plus ~3 bytes per token for newer messages) and returns the output
// budget.
func (a *Agent) maxTokens(st *runState, history []Message) (int, error) {
	out := a.cfg.MaxOutputTokens
	if a.cfg.ContextLength <= 0 {
		return out, nil
	}
	used, afterSeq, ok, err := lastCallSize(st.db, a.store, a.sessionID)
	if err != nil {
		return 0, err
	}
	est := int(used)
	if !ok {
		est = len(a.system) / 3
		for _, t := range a.tools {
			est += len(t) / 3
		}
		afterSeq = 0
	}
	for _, m := range history {
		if m.Seq > afterSeq {
			est += len(m.Raw)/3 + 4
		}
	}
	remaining := a.cfg.ContextLength - est
	if remaining <= 0 {
		return 0, fmt.Errorf("%w (estimated %d of %d tokens)", ErrContextLengthExceeded, est, a.cfg.ContextLength)
	}
	if out <= 0 || out > remaining {
		out = remaining
	}
	return out, nil
}
