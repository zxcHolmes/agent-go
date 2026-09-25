package agent

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"strings"
)

// Client is the entry point of the SDK. It holds the store and the shared
// configuration; everything else (sessions, messages, billing, agents) is
// reached through it. Create one per database at startup and share it.
type Client struct {
	cfg   Config
	store Store
	res   *resources // mounted docs and the system prompt file
	log   *slog.Logger
}

// NewClient validates cfg, creates the tables (unless cfg.SkipSchema) and
// recovers sessions left running by a crashed process (see
// Config.RecoverStaleAfter and ResetSession).
func NewClient(ctx context.Context, cfg Config) (*Client, error) {
	if cfg.Store == nil {
		return nil, errors.New("agent: Config.Store is required")
	}
	res, err := loadResources(cfg)
	if err != nil {
		return nil, err
	}
	c := &Client{cfg: cfg, store: cfg.Store, res: res, log: newLogger(cfg)}
	if !cfg.SkipSchema {
		if err := migrate(ctx, c.store); err != nil {
			return nil, err
		}
	}
	if err := c.Recover(ctx); err != nil {
		return nil, err
	}
	return c, nil
}

// Recover resets sessions left "running" or "stopping" by a dead process, as
// NewClient does at startup. Each recovered session goes through ResetSession.
func (c *Client) Recover(ctx context.Context) error {
	ids, err := recoverCrashed(ctx, c.store, c.cfg.RecoverStaleAfter)
	for _, id := range ids {
		c.log.Info("recovered session left running by a dead process", "session", id)
	}
	return err
}

// Agent returns an agent for sessionID, creating a new session when it is
// empty (read it back with Agent.SessionID).
func (c *Client) Agent(ctx context.Context, sessionID string, opts AgentOptions) (*Agent, error) {
	return newAgent(ctx, c, sessionID, opts)
}

// Docs lists the mounted documents (Config.Docs), sorted by title.
func (c *Client) Docs() []Doc {
	return append([]Doc(nil), c.res.docs...)
}

// ---- sessions ----

// CreateSession creates a new idle session and returns its id. metadata is
// optional and stored as JSON (e.g. the owning user id).
func (c *Client) CreateSession(ctx context.Context, metadata map[string]any) (string, error) {
	return createSession(ctx, c.store, metadata)
}

// Session loads a session: status, last error, metadata.
func (c *Client) Session(ctx context.Context, sessionID string) (*Session, error) {
	return getSession(ctx, c.store, sessionID)
}

// ResetSession recovers a session whose run died with its process:
//   - assistant messages still "streaming" become "interrupted" (partial
//     content is kept and stays in the history);
//   - RPC calls that were running become "failed" with a CodeCrashed error;
//   - queued / approved calls that never started become "cancelled";
//   - calls awaiting confirmation are kept, unless the session was stopping;
//   - the session becomes idle (or waiting_confirmation) and last_error
//     records the recovery.
//
// Only call it when you know no run is active for the session; Stop is the
// safe way to end a live one.
func (c *Client) ResetSession(ctx context.Context, sessionID string) error {
	return resetSession(ctx, c.store, sessionID)
}

// Stop stops a session without building a full agent; see Agent.Stop.
func (c *Client) Stop(ctx context.Context, sessionID string) error {
	cfg := c.cfg.withDefaults()
	a := &Agent{client: c, cfg: cfg, store: c.store, sessionID: sessionID, methods: map[string]Method{}, res: c.res, log: c.log}
	for _, m := range cfg.Methods {
		a.methods[m.Name] = m
	}
	if _, err := getSession(ctx, c.store, sessionID); err != nil {
		return err
	}
	return a.Stop(ctx)
}

// PendingCalls returns the RPC calls of a session waiting for confirmation.
func (c *Client) PendingCalls(ctx context.Context, sessionID string) ([]RPCCall, error) {
	return pendingCalls(ctx, c.store, sessionID)
}

// ---- queue ----

// Enqueue queues a user message for a session; see Agent.Enqueue.
func (c *Client) Enqueue(ctx context.Context, sessionID, prompt string) (string, error) {
	if strings.TrimSpace(prompt) == "" {
		return "", errors.New("agent: empty prompt")
	}
	raw, err := userRaw(prompt)
	if err != nil {
		return "", err
	}
	return enqueue(ctx, c.store, sessionID, raw)
}

// EnqueueMessage queues a caller-built user message (e.g. multimodal content).
func (c *Client) EnqueueMessage(ctx context.Context, sessionID string, raw json.RawMessage) (string, error) {
	return enqueue(ctx, c.store, sessionID, raw)
}

// QueuedMessages lists the messages of a session still waiting in the queue.
func (c *Client) QueuedMessages(ctx context.Context, sessionID string) ([]QueuedMessage, error) {
	return queuedMessages(ctx, c.store, sessionID)
}

// CancelQueued removes a message from the queue if it has not been sent yet.
func (c *Client) CancelQueued(ctx context.Context, sessionID, queueID string) error {
	return cancelQueued(ctx, c.store, sessionID, queueID)
}

// ---- messages ----

// LatestMessages returns the newest limit messages in chronological order.
func (c *Client) LatestMessages(ctx context.Context, sessionID string, limit int) ([]Message, error) {
	return latestMessages(ctx, c.store, sessionID, limit)
}

// MessagesBefore returns up to limit messages older than messageID, in
// chronological order (scrolling back through history).
func (c *Client) MessagesBefore(ctx context.Context, sessionID, messageID string, limit int) ([]Message, error) {
	return messagesBefore(ctx, c.store, sessionID, messageID, limit)
}

// MessagesAfter returns up to limit messages newer than messageID, in
// chronological order (polling for new and still-changing messages).
func (c *Client) MessagesAfter(ctx context.Context, sessionID, messageID string, limit int) ([]Message, error) {
	return messagesAfter(ctx, c.store, sessionID, messageID, limit)
}

// Message loads one message.
func (c *Client) Message(ctx context.Context, sessionID, messageID string) (*Message, error) {
	return getMessage(ctx, c.store, sessionID, messageID)
}

// ---- usage & billing ----

// Usage sums token usage and credits over all LLM calls of a session.
func (c *Client) Usage(ctx context.Context, sessionID string) (*UsageSummary, error) {
	return sessionUsage(ctx, c.store, sessionID)
}

// ListUsage returns every LLM call of a session, oldest first.
func (c *Client) ListUsage(ctx context.Context, sessionID string) ([]UsageRecord, error) {
	return listUsage(ctx, c.store, sessionID)
}

// UnbilledUsage returns the LLM calls not marked billed yet, oldest first, and
// their total. sessionID "" means all sessions. Records claimed by a
// SettleUsage that has not completed (BillID set, BilledAt nil) are included.
func (c *Client) UnbilledUsage(ctx context.Context, sessionID string) (*UsageSummary, []UsageRecord, error) {
	return unbilledUsage(ctx, c.store, sessionID)
}

// MarkBilled marks LLM call records as billed under billID (empty generates
// one) and returns the bill id. Records already billed are left unchanged.
func (c *Client) MarkBilled(ctx context.Context, billID string, recordIDs ...string) (string, error) {
	return markBilled(ctx, c.store, billID, recordIDs...)
}

// SettleUsage bills every unbilled LLM call of a session in one step, safely
// against concurrent settlements:
//
//  1. it atomically claims the session's unclaimed, unbilled records under a
//     new bill id (a record can only be claimed once);
//  2. it calls charge with the bill (e.g. deduct Bill.Summary.Cost.Total from
//     the user's balance, keyed by Bill.ID);
//  3. if charge succeeds the records are marked billed; if it fails they are
//     released for the next attempt and the error is returned.
//
// It returns (nil, nil) without calling charge when there is nothing to bill.
// If the process dies during charge, the records stay claimed (BillID set,
// BilledAt nil, visible in UnbilledUsage): check your ledger for the bill id,
// then call CompleteBill or ReleaseBill.
func (c *Client) SettleUsage(ctx context.Context, sessionID string, charge func(ctx context.Context, bill *Bill) error) (*Bill, error) {
	return settleUsage(ctx, c.store, sessionID, charge)
}

// CompleteBill marks all records claimed under billID as billed.
func (c *Client) CompleteBill(ctx context.Context, billID string) error {
	return completeBill(ctx, c.store, billID)
}

// ReleaseBill returns records claimed under billID but not billed to the
// unbilled pool, so the next SettleUsage picks them up again.
func (c *Client) ReleaseBill(ctx context.Context, billID string) error {
	return releaseBill(ctx, c.store, billID)
}
