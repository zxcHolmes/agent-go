package agent

import (
	"encoding/json"
	"errors"
	"time"
)

// Status is the lifecycle state of a session.
type Status string

const (
	// StatusIdle means no run is in progress; Chat and Continue may be called.
	StatusIdle Status = "idle"
	// StatusRunning means the agent loop is executing.
	StatusRunning Status = "running"
	// StatusStopping means Stop was called and the run is shutting down; it
	// becomes StatusIdle as soon as the run has exited.
	StatusStopping Status = "stopping"
	// StatusWaitingConfirmation means the loop paused on RPC calls that need
	// approval; call Confirm to resume.
	StatusWaitingConfirmation Status = "waiting_confirmation"
)

// CallStatus is the state of a single RPC call requested by the model.
type CallStatus string

const (
	CallQueued               CallStatus = "queued"
	CallAwaitingConfirmation CallStatus = "awaiting_confirmation"
	CallApproved             CallStatus = "approved"
	CallRunning              CallStatus = "running"
	CallRejected             CallStatus = "rejected"
	CallDone                 CallStatus = "done"
	CallCancelled            CallStatus = "cancelled"
	CallFailed               CallStatus = "failed" // the process crashed while the call was running
)

func (s CallStatus) finished() bool {
	switch s {
	case CallRejected, CallDone, CallCancelled, CallFailed:
		return true
	}
	return false
}

// MessageStatus tells polling clients whether a message may still change.
type MessageStatus string

const (
	// MessageStreaming: assistant message still being generated; content
	// is flushed to the store every Config.StreamFlushInterval.
	MessageStreaming MessageStatus = "streaming"
	// MessagePending: tool message whose call has not started yet (queued or
	// awaiting confirmation).
	MessagePending MessageStatus = "pending"
	// MessageRunning: tool message whose call is executing.
	MessageRunning MessageStatus = "running"
	// MessageDone: final.
	MessageDone MessageStatus = "done"
	// MessageInterrupted: assistant output cut off by Stop, an error or a
	// crash. Final; the partial content is kept in the history.
	MessageInterrupted MessageStatus = "interrupted"
	// MessageExcluded: kept for display but never sent to the model again,
	// e.g. a view_image message whose URL the provider could not load.
	MessageExcluded MessageStatus = "excluded"
)

// MessageKindViewImage marks the user message the SDK injects to show the
// model an image requested with the view_image tool. Its ToolCallID is the
// id of that tool call, so a UI can render it as an attachment rather than
// as something the user typed.
const MessageKindViewImage = "view_image"

// MessageKindCompaction marks where the context was compacted (see
// Config.Compaction). Messages before its RefSeq are no longer sent to the
// model. In summary mode its content is the summary, sent to the model in
// place of the dropped messages (status done); in anchor mode it is only a
// marker for the UI (status excluded).
const MessageKindCompaction = "compaction"

// Final reports whether the message will not change anymore.
func (s MessageStatus) Final() bool {
	return s == MessageDone || s == MessageInterrupted || s == MessageExcluded
}

// StopReason explains why a run returned.
type StopReason string

const (
	StopCompleted           StopReason = "completed"
	StopWaitingConfirmation StopReason = "waiting_confirmation"
	StopStopped             StopReason = "stopped"
	StopMaxSteps            StopReason = "max_steps"
)

var (
	ErrSessionNotFound       = errors.New("agent: session not found")
	ErrMessageNotFound       = errors.New("agent: message not found")
	ErrBusy                  = errors.New("agent: session is already running")
	ErrWaitingConfirmation   = errors.New("agent: session is waiting for rpc call confirmation")
	ErrNoPendingCalls        = errors.New("agent: session has no rpc calls waiting for confirmation")
	ErrCallNotPending        = errors.New("agent: rpc call is not waiting for confirmation")
	ErrStopped               = errors.New("agent: stopped")
	ErrLockLost              = errors.New("agent: session was taken over by another run")
	ErrContextLengthExceeded = errors.New("agent: conversation exceeds the model context length")
	ErrQueuedMessageSent     = errors.New("agent: queued message was already sent to the model")
	ErrInvalidUserMessage    = errors.New("agent: invalid user message")
)

// Session is a persisted conversation.
type Session struct {
	ID        string          `json:"id"`
	Status    Status          `json:"status"`
	LastError string          `json:"last_error,omitempty"`
	Metadata  json.RawMessage `json:"metadata,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
	UpdatedAt time.Time       `json:"updated_at"`
}

// Message is one chat message. Raw holds the exact JSON sent to the model; once
// final it never changes and is replayed byte-for-byte so provider prefix
// caches stay valid. Content, ToolCalls and ToolCallID are decoded from Raw.
type Message struct {
	ID        string `json:"id"`
	SessionID string `json:"session_id"`
	Seq       int64  `json:"seq"`
	Role      string `json:"role"`
	Kind      string `json:"kind,omitempty"` // "" for regular messages, or a MessageKind* constant
	// RefSeq is set on compaction messages: the seq of the first message the
	// model still sees verbatim after this compaction.
	RefSeq     int64           `json:"ref_seq,omitempty"`
	Status     MessageStatus   `json:"status"`
	Content    string          `json:"content"`
	Reasoning  string          `json:"reasoning,omitempty"` // streamed reasoning / thinking text
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
	Raw        json.RawMessage `json:"raw"`
	CreatedAt  time.Time       `json:"created_at"`
	UpdatedAt  time.Time       `json:"updated_at"`
}

// ToolCall is a tool call emitted by the model.
type ToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// RPCCall is a JSON-RPC call requested by the model, with its execution state.
type RPCCall struct {
	ID              string          `json:"id"`
	SessionID       string          `json:"session_id"`
	MessageID       string          `json:"message_id"`
	ToolCallID      string          `json:"tool_call_id"`
	Index           int             `json:"index"`
	Method          string          `json:"method"`
	Params          json.RawMessage `json:"params,omitempty"`
	RPCID           json.RawMessage `json:"rpc_id,omitempty"`
	RequireConfirm  bool            `json:"require_confirm"`
	Status          CallStatus      `json:"status"`
	Result          json.RawMessage `json:"result,omitempty"`            // JSON-RPC response object
	ResultMessageID string          `json:"result_message_id,omitempty"` // the tool message carrying Result
	CreatedAt       time.Time       `json:"created_at"`
	UpdatedAt       time.Time       `json:"updated_at"`
}

// Decision approves or rejects one call awaiting confirmation.
type Decision struct {
	CallID  string
	Approve bool
	Reason  string // returned to the model on rejection
}

// Approve builds an approving Decision.
func Approve(callID string) Decision { return Decision{CallID: callID, Approve: true} }

// Reject builds a rejecting Decision.
func Reject(callID, reason string) Decision { return Decision{CallID: callID, Reason: reason} }

// RunResult summarises one Chat / Continue / Confirm run.
type RunResult struct {
	SessionID    string     `json:"session_id"`
	Status       Status     `json:"status"`
	StopReason   StopReason `json:"stop_reason,omitempty"`
	FinishReason string     `json:"finish_reason,omitempty"` // of the last LLM response
	Messages     []Message  `json:"messages"`                // messages appended by this run
	PendingCalls []RPCCall  `json:"pending_calls,omitempty"`
	Usage        Usage      `json:"usage"`
	Cost         Cost       `json:"cost"`
}

// Reply returns the text of the last assistant message produced by the run.
func (r *RunResult) Reply() string {
	for i := len(r.Messages) - 1; i >= 0; i-- {
		if r.Messages[i].Role == "assistant" && r.Messages[i].Content != "" {
			return r.Messages[i].Content
		}
	}
	return ""
}
