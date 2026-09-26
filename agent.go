package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
)

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
	res           *resources
	log           *slog.Logger
}

func newAgent(ctx context.Context, c *Client, sessionID string, opts AgentOptions) (*Agent, error) {
	cfg := c.cfg
	cfg.Methods = append([]Method(nil), c.cfg.Methods...)
	if opts.Override != nil {
		opts.Override(&cfg)
	}
	cfg.Store = c.store
	cfg = cfg.withDefaults()
	res := c.res
	if opts.Override != nil { // the override may point at other files
		var err error
		if res, err = loadResources(cfg); err != nil {
			return nil, err
		}
	}
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
		res:           res,
		log:           newLogger(cfg),
		llm: &llmClient{
			endpoint:    chatEndpoint(cfg.BaseURL),
			apiKey:      cfg.APIKey,
			headers:     cfg.Headers,
			http:        httpClient,
			maxRetries:  cfg.MaxRetries,
			idleTimeout: cfg.StreamIdleTimeout,
			log:         newLogger(cfg),
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
	if cfg.ToolName == ViewImageTool || cfg.ToolName == ReadDocTool {
		return nil, fmt.Errorf("agent: ToolName %q clashes with a built-in tool", cfg.ToolName)
	}
	if len(a.methods) > 0 {
		tool, err := a.buildTool()
		if err != nil {
			return nil, err
		}
		a.tools = append(a.tools, tool)
	}
	if len(res.docs) > 0 {
		tool, err := a.buildReadDocTool()
		if err != nil {
			return nil, err
		}
		a.tools = append(a.tools, tool)
	}
	if cfg.ViewImage {
		tool, err := a.buildViewImageTool()
		if err != nil {
			return nil, err
		}
		a.tools = append(a.tools, tool)
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
//
// Cancelling ctx does not stop a run that has started (only Stop does), so
// running Chat in a goroutine with an HTTP request's ctx is safe; ctx only
// matters until the session is acquired. The same holds for ChatMessage,
// Continue and Confirm.
func (a *Agent) Chat(ctx context.Context, prompt string) (*RunResult, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("agent: empty prompt")
	}
	raw, err := userRaw(prompt)
	if err != nil {
		return nil, err
	}
	return a.ChatMessage(ctx, raw)
}

// ChatMessage is Chat with a caller-built user message, e.g. multimodal
// content: {"role":"user","content":[{"type":"text","text":"..."},{"type":"image_url",...}]}.
// Only text and image_url parts are accepted, and images must be http(s)
// URLs (no base64 data URLs); otherwise it returns ErrInvalidUserMessage.
func (a *Agent) ChatMessage(ctx context.Context, raw json.RawMessage) (*RunResult, error) {
	if err := checkUserMessage(raw); err != nil {
		return nil, err
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
		// Messages queued while the session was idle come before the new prompt.
		if _, err := a.drainQueue(ctx, st); err != nil {
			return err
		}
		m, err := a.userMessage(raw)
		if err != nil {
			return err
		}
		return a.insertMessage(ctx, st, &m)
	}, true)
}

// Enqueue adds a user message to the conversation without waiting for the
// current run: it joins right before the agent's next LLM call (or at the
// start of the next Chat / Continue if the session is idle). Use it to let
// users add details while the agent is working. It returns the queue id.
func (a *Agent) Enqueue(ctx context.Context, prompt string) (string, error) {
	return a.client.Enqueue(ctx, a.sessionID, prompt)
}

// CancelQueued withdraws a queued message before it is sent; see
// Client.CancelQueued.
func (a *Agent) CancelQueued(ctx context.Context, queueID string) error {
	return a.client.CancelQueued(ctx, a.sessionID, queueID)
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
				err = setCallStatus(st.db, a.store, st.lease, c, CallApproved)
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
