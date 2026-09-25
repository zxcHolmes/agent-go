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

// Config configures an Agent.
type Config struct {
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

	// SessionID resumes an existing session. Empty creates a new one.
	SessionID string
	// SystemPrompt is prepended to every request. Keep it stable across calls
	// so the provider prefix cache keeps hitting.
	SystemPrompt string
	// RPCDoc is free-form documentation of your JSON-RPC API, appended to the
	// system prompt after the auto-generated method list.
	RPCDoc string
	// Methods are the RPC methods the model may call.
	Methods []Method
	// ContextParams are passed to every handler through Call.ContextParams
	// (e.g. user id, tenant, auth token). They are not sent to the model and
	// not persisted.
	ContextParams map[string]any

	// Store persists sessions, messages, rpc calls and usage. Required.
	Store Store
	// Billing prices every LLM call in credits. Optional.
	Billing Billing

	// ToolName is the name of the single JSON-RPC tool (default "json_rpc").
	ToolName string
	// MaxSteps bounds LLM calls per run (default 50).
	MaxSteps int
	// ExtraBody is merged into every request body (temperature, top_p,
	// reasoning_effort, ...). It cannot override model, messages or tools.
	ExtraBody map[string]any
	// Headers are added to every LLM request.
	Headers    map[string]string
	HTTPClient *http.Client
	// MaxRetries for 429/5xx/network errors (default 2, negative disables).
	MaxRetries int
	// StaleAfter is how long a "running" session may go without a heartbeat
	// before another run may take it over, e.g. after a crash (default 15m).
	StaleAfter time.Duration

	// OnMessage is called after each message is persisted.
	OnMessage func(ctx context.Context, m Message)
	// OnUsage is called after each LLM call is recorded, e.g. to deduct credits.
	OnUsage func(ctx context.Context, r UsageRecord)
}

// Agent runs the tool-calling loop for one session. It is safe for concurrent
// use; at most one run per session executes at a time (across processes too,
// through the session row lock).
type Agent struct {
	cfg       Config
	store     Store
	llm       *llmClient
	methods   map[string]Method
	system    json.RawMessage
	tools     []json.RawMessage
	sessionID string
}

// New creates an agent bound to cfg.SessionID, creating a new session when it
// is empty.
func New(ctx context.Context, cfg Config) (*Agent, error) {
	if cfg.Store == nil {
		return nil, errors.New("agent: Config.Store is required")
	}
	if cfg.BaseURL == "" || cfg.Model == "" {
		return nil, errors.New("agent: Config.BaseURL and Config.Model are required")
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
		cfg.StaleAfter = 15 * time.Minute
	}
	httpClient := cfg.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 10 * time.Minute}
	}
	a := &Agent{
		cfg:     cfg,
		store:   cfg.Store,
		methods: make(map[string]Method, len(cfg.Methods)),
		llm: &llmClient{
			endpoint:   chatEndpoint(cfg.BaseURL),
			apiKey:     cfg.APIKey,
			headers:    cfg.Headers,
			http:       httpClient,
			maxRetries: cfg.MaxRetries,
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

	if cfg.SessionID == "" {
		if a.sessionID, err = CreateSession(ctx, a.store, nil); err != nil {
			return nil, err
		}
	} else {
		if _, err := GetSession(ctx, a.store, cfg.SessionID); err != nil {
			return nil, err
		}
		a.sessionID = cfg.SessionID
	}
	return a, nil
}

// SessionID returns the id of the session this agent drives.
func (a *Agent) SessionID() string { return a.sessionID }

// Session loads the session row.
func (a *Agent) Session(ctx context.Context) (*Session, error) {
	return GetSession(ctx, a.store, a.sessionID)
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
	return PendingCalls(ctx, a.store, a.sessionID)
}

// LatestMessages returns the newest limit messages of the session.
func (a *Agent) LatestMessages(ctx context.Context, limit int) ([]Message, error) {
	return LatestMessages(ctx, a.store, a.sessionID, limit)
}

// MessagesBefore returns up to limit messages older than messageID.
func (a *Agent) MessagesBefore(ctx context.Context, messageID string, limit int) ([]Message, error) {
	return MessagesBefore(ctx, a.store, a.sessionID, messageID, limit)
}

// MessagesAfter returns up to limit messages newer than messageID.
func (a *Agent) MessagesAfter(ctx context.Context, messageID string, limit int) ([]Message, error) {
	return MessagesAfter(ctx, a.store, a.sessionID, messageID, limit)
}

// Usage sums token usage and credits of the session.
func (a *Agent) Usage(ctx context.Context) (*UsageSummary, error) {
	return SessionUsage(ctx, a.store, a.sessionID)
}

// Chat appends a user message and runs the agent loop until the model answers
// without tool calls, a call needs confirmation, Stop is called, or MaxSteps
// is reached. It blocks for the whole run.
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
		open, err := a.openCalls(st)
		if err != nil {
			return err
		}
		for _, c := range open {
			if c.Status == CallAwaitingConfirmation {
				return ErrWaitingConfirmation
			}
		}
		// Leftovers from an interrupted run: close them so the history stays valid.
		if err := a.cancelCalls(ctx, st, open, "cancelled: superseded by a new user message"); err != nil {
			return err
		}
		_, err = a.appendMessage(ctx, st, raw)
		return err
	})
}

// Continue resumes the loop without a new user message, e.g. after Stop, an
// error, or a truncated answer. Unfinished RPC calls are executed first.
func (a *Agent) Continue(ctx context.Context) (*RunResult, error) {
	return a.run(ctx, []Status{StatusIdle}, nil)
}

// Confirm approves or rejects calls awaiting confirmation. Once no call is
// awaiting anymore, the loop continues and Confirm blocks like Chat.
func (a *Agent) Confirm(ctx context.Context, decisions ...Decision) (*RunResult, error) {
	if len(decisions) == 0 {
		return nil, errors.New("agent: no decisions")
	}
	return a.run(ctx, []Status{StatusWaitingConfirmation}, func(ctx context.Context, st *runState) error {
		pending, err := PendingCalls(st.db, a.store, a.sessionID)
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
				c.Status = CallApproved
			} else {
				msg := "rejected by user"
				if d.Reason != "" {
					msg += ": " + d.Reason
				}
				c.Status = CallRejected
				c.Result = rpcErrorResponse(c.RPCID, &RPCError{Code: CodeRejected, Message: msg})
			}
			if err := updateCall(st.db, a.store, c); err != nil {
				return err
			}
		}
		return nil
	})
}

// Stop interrupts the running loop of this session, in this process or
// another one (the loop checks a stop flag before every step). Calls not yet
// executed are cancelled; the session becomes idle (or waiting_confirmation
// if calls still await approval). Stop returns immediately.
func (a *Agent) Stop(ctx context.Context) error {
	if v, ok := running.Load(a.sessionID); ok {
		v.(*localRun).cancel(ErrStopped)
	}
	_, err := a.store.Exec(ctx, "UPDATE agent_sessions SET stop_requested = 1 WHERE id = ? AND status = ?", a.sessionID, string(StatusRunning))
	return err
}

// ---- run machinery ----

var running sync.Map // session id -> *localRun

type localRun struct {
	runID  string
	cancel context.CancelCauseFunc
}

type runState struct {
	runID  string
	db     context.Context // not cancelled by Stop, so bookkeeping always completes
	cancel context.CancelCauseFunc
	res    *RunResult
}

func (a *Agent) run(ctx context.Context, from []Status, prepare func(context.Context, *runState) error) (*RunResult, error) {
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
	var reason StopReason
	if prepare != nil {
		err = prepare(runCtx, st)
	}
	if err == nil {
		reason, err = a.loop(runCtx, st)
	}
	return a.finish(runCtx, st, reason, err)
}

func (a *Agent) finish(runCtx context.Context, st *runState, reason StopReason, err error) (*RunResult, error) {
	res := st.res
	if errors.Is(err, ErrLockLost) {
		return res, err
	}
	interrupted := err != nil && runCtx.Err() != nil
	stopped := interrupted && errors.Is(context.Cause(runCtx), ErrStopped)
	if stopped {
		// A requested stop is a normal outcome, not an error.
		err = nil
	}
	var bookErr error // bookkeeping failures while wrapping up
	if interrupted {
		open, oerr := openCalls(st.db, a.store, a.sessionID)
		if oerr == nil {
			oerr = a.cancelCalls(runCtx, st, open, "cancelled: the agent was stopped before this call ran")
		}
		bookErr = errors.Join(bookErr, oerr)
	}
	pending, perr := PendingCalls(st.db, a.store, a.sessionID)
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
	if bookErr != nil {
		err = errors.Join(err, bookErr)
	}
	res.Status, res.PendingCalls = status, pending
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

// acquire atomically moves the session to running if it is in one of the
// allowed states, or if a previous run stopped heartbeating (crash).
func (a *Agent) acquire(ctx context.Context, from []Status) (string, error) {
	runID := newID("run")
	now := nowMillis()
	args := []any{string(StatusRunning), runID, now, a.sessionID}
	for _, s := range from {
		args = append(args, string(s))
	}
	args = append(args, string(StatusRunning), now-a.cfg.StaleAfter.Milliseconds())
	ph := strings.TrimSuffix(strings.Repeat("?, ", len(from)), ", ")
	if _, err := a.store.Exec(ctx,
		"UPDATE agent_sessions SET status = ?, run_id = ?, stop_requested = 0, last_error = '', updated_at = ? WHERE id = ? AND (status IN ("+ph+") OR (status = ? AND updated_at < ?))",
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
	s, err := GetSession(ctx, a.store, a.sessionID)
	if err != nil {
		return "", err
	}
	switch s.Status {
	case StatusRunning:
		return "", ErrBusy
	case StatusWaitingConfirmation:
		return "", ErrWaitingConfirmation
	default:
		return "", ErrNoPendingCalls
	}
}

func (a *Agent) sessionLock(ctx context.Context) (runID string, stop bool, err error) {
	rows, err := a.store.Query(ctx, "SELECT run_id, stop_requested FROM agent_sessions WHERE id = ?", a.sessionID)
	if err != nil {
		return "", false, err
	}
	defer rows.Close()
	if !rows.Next() {
		if err := rows.Err(); err != nil {
			return "", false, err
		}
		return "", false, ErrSessionNotFound
	}
	var flag int64
	err = rows.Scan(&runID, &flag)
	return runID, flag != 0, err
}

// checkpoint heartbeats the lock and honours stop requests.
func (a *Agent) checkpoint(ctx context.Context, st *runState) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	if _, err := a.store.Exec(st.db, "UPDATE agent_sessions SET updated_at = ? WHERE id = ? AND run_id = ?", nowMillis(), a.sessionID, st.runID); err != nil {
		return err
	}
	owner, stop, err := a.sessionLock(st.db)
	if err != nil {
		return err
	}
	if owner != st.runID {
		return ErrLockLost
	}
	if stop {
		st.cancel(ErrStopped)
		return ErrStopped
	}
	return nil
}

func (a *Agent) release(st *runState, status Status, lastErr string) error {
	_, err := a.store.Exec(st.db, "UPDATE agent_sessions SET status = ?, run_id = '', stop_requested = 0, last_error = ?, updated_at = ? WHERE id = ? AND run_id = ?",
		string(status), lastErr, nowMillis(), a.sessionID, st.runID)
	return err
}

func (a *Agent) loop(ctx context.Context, st *runState) (StopReason, error) {
	llmCalls := 0
	for {
		if err := a.checkpoint(ctx, st); err != nil {
			return "", err
		}
		open, err := a.openCalls(st)
		if err != nil {
			return "", err
		}
		if len(open) > 0 {
			waiting := false
			for i := range open {
				c := &open[i]
				switch c.Status {
				case CallQueued, CallApproved:
					if err := a.checkpoint(ctx, st); err != nil {
						return "", err
					}
					c.Result = a.invoke(ctx, c)
					c.Status = CallDone
					if err := updateCall(st.db, a.store, c); err != nil {
						return "", err
					}
				case CallAwaitingConfirmation:
					waiting = true
				}
			}
			if waiting {
				return StopWaitingConfirmation, nil
			}
			if err := a.flushResults(ctx, st, open); err != nil {
				return "", err
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

// openCalls returns calls without a written tool result. If the last message
// requested tool calls that were never recorded (crash between writes), it
// records them first.
func (a *Agent) openCalls(st *runState) ([]RPCCall, error) {
	open, err := openCalls(st.db, a.store, a.sessionID)
	if err != nil || len(open) > 0 {
		return open, err
	}
	last, err := lastMessage(st.db, a.store, a.sessionID)
	if err != nil || last == nil || last.Role != "assistant" || len(last.ToolCalls) == 0 {
		return nil, err
	}
	existing, err := callsForMessage(st.db, a.store, last.ID)
	if err != nil || len(existing) > 0 {
		return nil, err
	}
	for i, tc := range last.ToolCalls {
		c := a.newCall(last, i, tc)
		if err := insertCall(st.db, a.store, &c); err != nil {
			return nil, err
		}
	}
	return openCalls(st.db, a.store, a.sessionID)
}

func (a *Agent) newCall(m *Message, i int, tc ToolCall) RPCCall {
	now := time.Now()
	c := RPCCall{
		ID: newID("call"), SessionID: a.sessionID, MessageID: m.ID, ToolCallID: tc.ID, Index: i,
		Method: tc.Function.Name, Status: CallQueued, CreatedAt: now, UpdatedAt: now,
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

// cancelCalls closes not-yet-executed calls with a cancellation error and, if
// nothing awaits confirmation, writes their tool messages.
func (a *Agent) cancelCalls(ctx context.Context, st *runState, open []RPCCall, reason string) error {
	if len(open) == 0 {
		return nil
	}
	waiting := false
	for i := range open {
		c := &open[i]
		switch c.Status {
		case CallQueued, CallApproved:
			c.Status = CallCancelled
			c.Result = rpcErrorResponse(c.RPCID, &RPCError{Code: CodeCancelled, Message: reason})
			if err := updateCall(st.db, a.store, c); err != nil {
				return err
			}
		case CallAwaitingConfirmation:
			waiting = true
		}
	}
	if waiting {
		return nil
	}
	return a.flushResults(ctx, st, open)
}

// flushResults writes one tool message per call, in the model's call order.
func (a *Agent) flushResults(ctx context.Context, st *runState, calls []RPCCall) error {
	for i := range calls {
		c := &calls[i]
		raw, err := marshalJSON(struct {
			Role       string `json:"role"`
			ToolCallID string `json:"tool_call_id"`
			Content    string `json:"content"`
		}{"tool", c.ToolCallID, string(c.Result)})
		if err != nil {
			return err
		}
		m, err := a.appendMessage(ctx, st, raw)
		if err != nil {
			return err
		}
		c.ResultMessageID = m.ID
		if err := updateCall(st.db, a.store, c); err != nil {
			return err
		}
	}
	return nil
}

func (a *Agent) appendMessage(ctx context.Context, st *runState, raw json.RawMessage) (Message, error) {
	m, err := insertMessage(st.db, a.store, a.sessionID, raw)
	if err != nil {
		return m, err
	}
	st.res.Messages = append(st.res.Messages, m)
	if a.cfg.OnMessage != nil {
		a.cfg.OnMessage(ctx, m)
	}
	return m, nil
}

// step performs one LLM call and records its message and usage.
func (a *Agent) step(ctx context.Context, st *runState) (bool, error) {
	history, err := allMessages(st.db, a.store, a.sessionID)
	if err != nil {
		return false, err
	}
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

	start := time.Now()
	resp, err := a.llm.complete(ctx, req, a.cfg.ExtraBody)
	if err != nil {
		return false, err
	}
	latency := time.Since(start).Milliseconds()
	choice := resp.Choices[0]

	m, err := a.appendMessage(ctx, st, choice.Message)
	if err != nil {
		return false, err
	}
	usage := parseUsage(resp.Usage)
	var cost Cost
	if a.cfg.Billing != nil {
		cost = a.cfg.Billing.Cost(a.cfg.Model, usage)
	}
	rec := UsageRecord{
		ID: newID("llm"), SessionID: a.sessionID, MessageID: m.ID, Model: a.cfg.Model,
		ResponseID: resp.ID, FinishReason: choice.FinishReason, Usage: usage, RawUsage: resp.Usage,
		Cost: cost, LatencyMs: latency, CreatedAt: time.Now(),
	}
	if err := insertUsage(st.db, a.store, &rec); err != nil {
		return false, err
	}
	st.res.Usage.add(usage)
	st.res.Cost.add(cost)
	st.res.FinishReason = choice.FinishReason
	if a.cfg.OnUsage != nil {
		a.cfg.OnUsage(ctx, rec)
	}
	for i, tc := range m.ToolCalls {
		c := a.newCall(&m, i, tc)
		if err := insertCall(st.db, a.store, &c); err != nil {
			return false, err
		}
	}
	return len(m.ToolCalls) > 0, nil
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
