package agent

import (
	"encoding/json"
	"time"
)

// Usage holds every token counter reported by OpenAI-compatible providers.
// PromptTokens includes CachedTokens and CacheWriteTokens, as in the OpenAI API.
type Usage struct {
	PromptTokens             int64 `json:"prompt_tokens"`
	CompletionTokens         int64 `json:"completion_tokens"`
	TotalTokens              int64 `json:"total_tokens"`
	CachedTokens             int64 `json:"cached_tokens"`      // prompt tokens read from cache
	CacheWriteTokens         int64 `json:"cache_write_tokens"` // prompt tokens written to cache
	ReasoningTokens          int64 `json:"reasoning_tokens"`
	AudioInputTokens         int64 `json:"audio_input_tokens"`
	AudioOutputTokens        int64 `json:"audio_output_tokens"`
	AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
}

func (u *Usage) add(o Usage) {
	u.PromptTokens += o.PromptTokens
	u.CompletionTokens += o.CompletionTokens
	u.TotalTokens += o.TotalTokens
	u.CachedTokens += o.CachedTokens
	u.CacheWriteTokens += o.CacheWriteTokens
	u.ReasoningTokens += o.ReasoningTokens
	u.AudioInputTokens += o.AudioInputTokens
	u.AudioOutputTokens += o.AudioOutputTokens
	u.AcceptedPredictionTokens += o.AcceptedPredictionTokens
	u.RejectedPredictionTokens += o.RejectedPredictionTokens
}

// parseUsage understands the OpenAI shape plus common vendor variants
// (DeepSeek prompt_cache_hit_tokens, Anthropic-style cache_*_input_tokens).
func parseUsage(raw json.RawMessage) Usage {
	var u struct {
		PromptTokens        int64 `json:"prompt_tokens"`
		CompletionTokens    int64 `json:"completion_tokens"`
		TotalTokens         int64 `json:"total_tokens"`
		PromptTokensDetails struct {
			CachedTokens     int64 `json:"cached_tokens"`
			CacheWriteTokens int64 `json:"cache_write_tokens"`
			AudioTokens      int64 `json:"audio_tokens"`
		} `json:"prompt_tokens_details"`
		CompletionTokensDetails struct {
			ReasoningTokens          int64 `json:"reasoning_tokens"`
			AudioTokens              int64 `json:"audio_tokens"`
			AcceptedPredictionTokens int64 `json:"accepted_prediction_tokens"`
			RejectedPredictionTokens int64 `json:"rejected_prediction_tokens"`
		} `json:"completion_tokens_details"`
		PromptCacheHitTokens     int64 `json:"prompt_cache_hit_tokens"`
		CacheReadInputTokens     int64 `json:"cache_read_input_tokens"`
		CacheCreationInputTokens int64 `json:"cache_creation_input_tokens"`
	}
	if len(raw) == 0 || json.Unmarshal(raw, &u) != nil {
		return Usage{}
	}
	out := Usage{
		PromptTokens:             u.PromptTokens,
		CompletionTokens:         u.CompletionTokens,
		TotalTokens:              u.TotalTokens,
		CachedTokens:             firstNonZero(u.PromptTokensDetails.CachedTokens, u.PromptCacheHitTokens, u.CacheReadInputTokens),
		CacheWriteTokens:         firstNonZero(u.PromptTokensDetails.CacheWriteTokens, u.CacheCreationInputTokens),
		ReasoningTokens:          u.CompletionTokensDetails.ReasoningTokens,
		AudioInputTokens:         u.PromptTokensDetails.AudioTokens,
		AudioOutputTokens:        u.CompletionTokensDetails.AudioTokens,
		AcceptedPredictionTokens: u.CompletionTokensDetails.AcceptedPredictionTokens,
		RejectedPredictionTokens: u.CompletionTokensDetails.RejectedPredictionTokens,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.PromptTokens + out.CompletionTokens
	}
	return out
}

func firstNonZero(vs ...int64) int64 {
	for _, v := range vs {
		if v != 0 {
			return v
		}
	}
	return 0
}

// Cost is the credit cost of one or more LLM calls.
type Cost struct {
	Input       float64 `json:"input"`        // uncached prompt tokens
	CachedInput float64 `json:"cached_input"` // cache reads
	CacheWrite  float64 `json:"cache_write"`
	Output      float64 `json:"output"`
	Total       float64 `json:"total"`
}

func (c *Cost) add(o Cost) {
	c.Input += o.Input
	c.CachedInput += o.CachedInput
	c.CacheWrite += o.CacheWrite
	c.Output += o.Output
	c.Total += o.Total
}

// Billing prices the usage of one LLM call.
type Billing interface {
	Cost(model string, usage Usage) Cost
}

// Pricing is a Billing provider with prices in credits per 1,000,000 tokens.
// Example: Pricing{Input: 100, Output: 1000, CacheRead: 10}.
type Pricing struct {
	Input      float64 `json:"input"`
	Output     float64 `json:"output"`
	CacheRead  float64 `json:"cache_read"`
	CacheWrite float64 `json:"cache_write"` // 0 means same as Input
}

func (p Pricing) Cost(_ string, u Usage) Cost {
	uncached := u.PromptTokens - u.CachedTokens - u.CacheWriteTokens
	if uncached < 0 {
		uncached = 0
	}
	writePrice := p.CacheWrite
	if writePrice == 0 {
		writePrice = p.Input
	}
	c := Cost{
		Input:       float64(uncached) * p.Input / 1e6,
		CachedInput: float64(u.CachedTokens) * p.CacheRead / 1e6,
		CacheWrite:  float64(u.CacheWriteTokens) * writePrice / 1e6,
		Output:      float64(u.CompletionTokens) * p.Output / 1e6,
	}
	c.Total = c.Input + c.CachedInput + c.CacheWrite + c.Output
	return c
}

// PricingTable prices each model separately, falling back to Default.
type PricingTable struct {
	Default Pricing
	Models  map[string]Pricing
}

func (t PricingTable) Cost(model string, u Usage) Cost {
	if p, ok := t.Models[model]; ok {
		return p.Cost(model, u)
	}
	return t.Default.Cost(model, u)
}

// UsageRecord is the persisted record of one LLM call.
type UsageRecord struct {
	ID           string          `json:"id"`
	SessionID    string          `json:"session_id"`
	MessageID    string          `json:"message_id"` // assistant message produced by the call
	Model        string          `json:"model"`
	ResponseID   string          `json:"response_id"`
	FinishReason string          `json:"finish_reason"`
	Usage        Usage           `json:"usage"`
	RawUsage     json.RawMessage `json:"raw_usage,omitempty"` // usage object exactly as returned
	Cost         Cost            `json:"cost"`
	LatencyMs    int64           `json:"latency_ms"`
	CreatedAt    time.Time       `json:"created_at"`

	// Billing state. Unbilled: BilledAt == nil. BillID/ClaimedAt are set when
	// SettleUsage claims the record; BilledAt when it is marked billed.
	BillID    string     `json:"bill_id,omitempty"`
	ClaimedAt *time.Time `json:"claimed_at,omitempty"`
	BilledAt  *time.Time `json:"billed_at,omitempty"`
}

// UsageSummary aggregates all LLM calls of a session.
type UsageSummary struct {
	Calls int64 `json:"calls"`
	Usage Usage `json:"usage"`
	Cost  Cost  `json:"cost"`
}
