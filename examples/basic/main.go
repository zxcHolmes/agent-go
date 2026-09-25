// Command basic is a runnable demo of agent-go against a real
// OpenAI-compatible endpoint, storing everything in a local SQLite file.
//
//	OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... MODEL=gpt-4o-mini go run ./basic
package main

import (
	"bufio"
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"strings"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

type getOrderParams struct {
	OrderID string `json:"order_id"`
}

func (p getOrderParams) Validate() error {
	if !strings.HasPrefix(p.OrderID, "ORD-") {
		return errors.New(`order_id must look like "ORD-123"`)
	}
	return nil
}

type refundParams struct {
	OrderID string  `json:"order_id"`
	Amount  float64 `json:"amount"`
}

func main() {
	ctx := context.Background()

	db, err := sql.Open("sqlite", "file:agent.db?_pragma=busy_timeout(5000)")
	if err != nil {
		log.Fatal(err)
	}
	store := agent.NewSQLStore(db, agent.SQLite)
	if err := agent.Init(ctx, store); err != nil {
		log.Fatal(err)
	}

	a, err := agent.New(ctx, agent.Config{
		BaseURL:         os.Getenv("OPENAI_BASE_URL"),
		APIKey:          os.Getenv("OPENAI_API_KEY"),
		Model:           os.Getenv("MODEL"),
		ContextLength:   128000,
		MaxOutputTokens: 4096,
		SessionID:       os.Getenv("SESSION_ID"), // empty = new session
		SystemPrompt:    "You are a customer support agent. Be concise.",
		RPCDoc:          "Order ids look like ORD-123. Refund amounts are in USD.",
		ContextParams:   map[string]any{"user_id": "u_42"},
		Store:           store,
		Billing:         agent.Pricing{Input: 100, Output: 1000, CacheRead: 10},
		Methods: []agent.Method{
			{
				Name:        "get_order",
				Description: "Get an order of the current user",
				Params:      map[string]any{"type": "object", "properties": map[string]any{"order_id": map[string]any{"type": "string"}}, "required": []string{"order_id"}},
				Handler: agent.Typed(func(ctx context.Context, c *agent.Call, p getOrderParams) (any, error) {
					return map[string]any{"order_id": p.OrderID, "owner": c.Value("user_id"), "status": "delivered", "total": 59.9}, nil
				}),
			},
			{
				Name:           "refund_order",
				Description:    "Refund an order",
				RequireConfirm: true,
				Params:         map[string]any{"type": "object", "properties": map[string]any{"order_id": map[string]any{"type": "string"}, "amount": map[string]any{"type": "number"}}},
				Handler: agent.Typed(func(ctx context.Context, c *agent.Call, p refundParams) (any, error) {
					return map[string]any{"refunded": p.Amount}, nil
				}),
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("session:", a.SessionID())

	in := bufio.NewScanner(os.Stdin)
	for fmt.Print("> "); in.Scan(); fmt.Print("> ") {
		res, err := a.Chat(ctx, in.Text())
		for err == nil && res.Status == agent.StatusWaitingConfirmation {
			var ds []agent.Decision
			for _, c := range res.PendingCalls {
				fmt.Printf("approve %s %s? [y/N] ", c.Method, c.Params)
				in.Scan()
				if strings.EqualFold(strings.TrimSpace(in.Text()), "y") {
					ds = append(ds, agent.Approve(c.ID))
				} else {
					ds = append(ds, agent.Reject(c.ID, "user declined"))
				}
			}
			res, err = a.Confirm(ctx, ds...)
		}
		if err != nil {
			log.Println("error:", err)
			continue
		}
		fmt.Println(res.Reply())
		fmt.Printf("[tokens in=%d cached=%d out=%d, credits=%.4f]\n",
			res.Usage.PromptTokens, res.Usage.CachedTokens, res.Usage.CompletionTokens, res.Cost.Total)
	}
}
