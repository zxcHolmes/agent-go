# agent-go

一个轻量、零第三方依赖的 Go Agent SDK，对接任意 **OpenAI 兼容**的 Chat Completions 接口。

- 模型只有**一个工具**：JSON-RPC 2.0 调用（`method` / `params` / `id`）。你注册的 Go 方法都通过它被调用，SDK 负责路由、参数错误回传、panic 兜底。
- 会话、消息、RPC 调用、每次 LLM 调用的全部 token 字段和积分都持久化到**抽象的 SQL 存储层**，通过 session id 随时恢复对话。
- 消息按原始 JSON 字节完整保存并原样回放，保证大模型**前缀缓存**不丢失。
- 支持需要人工确认的 RPC 方法、Stop / Continue、跨进程的会话锁。

## 安装

```bash
go get github.com/zxcHolmes/agent-go
```

要求 Go 1.21+。包名是 `agent`，建议显式起别名：

```go
import agent "github.com/zxcHolmes/agent-go"
```

SDK 本身不依赖任何数据库驱动，按需自行引入，例如：

| 数据库 | 驱动 | Dialect |
| --- | --- | --- |
| SQLite | `modernc.org/sqlite`（纯 Go）或 `github.com/mattn/go-sqlite3` | `agent.SQLite` |
| PostgreSQL | `github.com/jackc/pgx/v5/stdlib` 或 `github.com/lib/pq` | `agent.Postgres` |
| MySQL | `github.com/go-sql-driver/mysql` | `agent.MySQL` |

## 快速开始

```go
package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"

	agent "github.com/zxcHolmes/agent-go"
	_ "modernc.org/sqlite"
)

type GetOrderParams struct {
	OrderID string `json:"order_id"`
}

// 可选：实现 Validate，校验失败会以 -32602 invalid params 返回给模型
func (p GetOrderParams) Validate() error {
	if p.OrderID == "" {
		return errors.New("order_id is required")
	}
	return nil
}

func main() {
	ctx := context.Background()

	// 1. 存储层 + 全局初始化（建表，幂等，启动时调用一次）
	db, _ := sql.Open("sqlite", "file:agent.db?_pragma=busy_timeout(5000)")
	store := agent.NewSQLStore(db, agent.SQLite)
	if err := agent.Init(ctx, store); err != nil {
		log.Fatal(err)
	}

	// 2. 创建会话，拿到 session id（可附带 metadata，比如所属用户）
	sessionID, _ := agent.CreateSession(ctx, store, map[string]any{"user_id": "u_42"})

	// 3. 创建 agent
	a, err := agent.New(ctx, agent.Config{
		BaseURL:         "https://api.openai.com/v1", // 任意 OpenAI 兼容地址
		APIKey:          "sk-...",
		Model:           "gpt-4o-mini",
		ContextLength:   128000, // 模型上下文长度
		MaxOutputTokens: 4096,   // 最大输出
		SessionID:       sessionID,
		SystemPrompt:    "你是一个客服助手。",
		RPCDoc:          "订单号格式为 ORD-123，金额单位为元。", // JSON-RPC 文档
		ContextParams:   map[string]any{"user_id": "u_42", "token": "..."}, // 身份等上下文，不会发给模型
		Store:           store,
		Billing:         agent.Pricing{Input: 100, Output: 1000, CacheRead: 10}, // 每百万 token 的积分
		Methods: []agent.Method{
			{
				Name:        "get_order",
				Description: "查询当前用户的订单",
				Params: map[string]any{ // 可选：JSON Schema，会写进系统提示词给模型看
					"type":       "object",
					"properties": map[string]any{"order_id": map[string]any{"type": "string"}},
					"required":   []string{"order_id"},
				},
				Handler: agent.Typed(func(ctx context.Context, call *agent.Call, p GetOrderParams) (any, error) {
					userID := call.Value("user_id") // 读取当前会话的上下文参数
					return map[string]any{"order_id": p.OrderID, "owner": userID, "status": "delivered"}, nil
				}),
			},
			{
				Name:           "refund_order",
				Description:    "给订单退款",
				RequireConfirm: true, // 需要确认才执行
				Handler: func(ctx context.Context, call *agent.Call) (any, error) {
					var p struct {
						OrderID string  `json:"order_id"`
						Amount  float64 `json:"amount"`
					}
					if err := call.Bind(&p); err != nil { // 严格解码，未知字段也会报错
						return nil, err
					}
					if p.Amount <= 0 {
						return nil, agent.InvalidParams("amount must be positive")
					}
					return map[string]any{"refunded": p.Amount}, nil
				},
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// 4. 对话
	res, err := a.Chat(ctx, "帮我查一下订单 ORD-1，然后退款 10 元")
	if err != nil {
		log.Fatal(err)
	}

	// 5. 有需要确认的方法时，agent 会暂停
	for res.Status == agent.StatusWaitingConfirmation {
		var ds []agent.Decision
		for _, c := range res.PendingCalls {
			fmt.Println("待确认:", c.Method, string(c.Params))
			ds = append(ds, agent.Approve(c.ID)) // 或 agent.Reject(c.ID, "理由")
		}
		if res, err = a.Confirm(ctx, ds...); err != nil {
			log.Fatal(err)
		}
	}

	fmt.Println(res.Reply())
	fmt.Printf("tokens: in=%d cached=%d out=%d credits=%.4f\n",
		res.Usage.PromptTokens, res.Usage.CachedTokens, res.Usage.CompletionTokens, res.Cost.Total)
}
```

完整可运行示例见 [`examples/basic`](examples/basic/main.go)：

```bash
cd examples
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... MODEL=gpt-4o-mini go run ./basic
```

## 概念与 API

### 全局初始化与会话

```go
agent.Init(ctx, store)                                  // 建表，幂等
agent.SchemaStatements(agent.Postgres)                  // 只要 DDL，交给自己的迁移工具
sid, _ := agent.CreateSession(ctx, store, metadata)     // 创建会话，返回 session id
s, _ := agent.GetSession(ctx, store, sid)               // 会话状态、last_error、metadata
agent.ResetSession(ctx, store, sid)                     // 进程崩溃后强制释放 running 状态
```

`agent.New` 的 `Config.SessionID` 为空时会自动创建新会话，用 `a.SessionID()` 取回。

### Agent 执行方法

| 方法 | 说明 |
| --- | --- |
| `a.Chat(ctx, prompt)` | 追加用户消息并运行 agent loop，阻塞直到结束 |
| `a.ChatMessage(ctx, rawJSON)` | 同上，自己构造用户消息（多模态图片等） |
| `a.Continue(ctx)` | 不加新消息继续运行（Stop 后、报错后、输出被截断后） |
| `a.Stop(ctx)` | 停止正在运行的 loop，立即返回；可在另一个 goroutine / 另一个进程调用 |
| `a.Confirm(ctx, decisions...)` | 批准 / 拒绝待确认的 RPC 调用，全部处理完后继续 loop |
| `a.Status(ctx)` | `idle` / `running` / `waiting_confirmation` |
| `a.PendingCalls(ctx)` | 待确认的 RPC 调用列表 |
| `a.Usage(ctx)` | 会话累计 token 与积分 |

一次运行（`Chat` / `Continue` / `Confirm`）会在以下情况返回，`RunResult.StopReason` 说明原因：

- `completed`：模型给出不含工具调用的回复
- `waiting_confirmation`：有 `RequireConfirm` 的方法等待确认（同一轮中不需要确认的调用已先执行）
- `stopped`：调用了 `Stop`，尚未执行的调用会以 `-32002 cancelled` 结果回填给模型，保证历史合法
- `max_steps`：达到 `Config.MaxSteps`（默认 50 次 LLM 调用）

`RunResult` 还包含本次新增的消息 `Messages`、`PendingCalls`、本次 `Usage` 和 `Cost`，`res.Reply()` 取最后一条助手文本。

**状态与并发**：每个会话同一时间只能有一个运行，通过数据库行锁实现（跨进程有效）。会话在运行中再调用 `Chat` 返回 `ErrBusy`；等待确认时调用 `Chat` / `Continue` 返回 `ErrWaitingConfirmation`。运行中每一步都会心跳，进程崩溃后超过 `Config.StaleAfter`（默认 15 分钟）其他运行可接管，也可以直接 `agent.ResetSession`。

典型 Web 服务用法：一个请求里 `go a.Chat(...)`，另一个请求用同一个 session id `agent.New(...)` 后调 `Stop` / `Status` / `PendingCalls` / `Confirm`。

### RPC 方法

模型拿到的唯一工具默认叫 `json_rpc`（`Config.ToolName` 可改），参数是一个 JSON-RPC 请求：

```json
{"jsonrpc": "2.0", "method": "get_order", "params": {"order_id": "ORD-1"}, "id": 1}
```

工具结果是 JSON-RPC 响应，原样作为 tool 消息回给模型：

```json
{"jsonrpc":"2.0","id":1,"result":{"order_id":"ORD-1","status":"delivered"}}
{"jsonrpc":"2.0","id":1,"error":{"code":-32602,"message":"order_id is required"}}
```

方法列表（名字、描述、参数 schema、是否需要确认）和 `RPCDoc` 会自动拼到系统提示词后面。

注册方法有两种写法：

```go
// 1. 泛型：自动解码 params 到结构体（严格模式，拒绝未知字段），可选 Validate() error
agent.Typed(func(ctx context.Context, call *agent.Call, p MyParams) (MyResult, error) { ... })

// 2. 原始 Handler：自己解码 / 校验
func(ctx context.Context, call *agent.Call) (any, error) {
	var p MyParams
	if err := call.Bind(&p); err != nil { return nil, err }
	...
}
```

`*agent.Call` 包含 `SessionID`、`CallID`、`Method`、`Params`（原始 JSON）、`ID` 和 `ContextParams`。也可以在任意深处用 `agent.CallFromContext(ctx)` 取到。

错误处理：

| 情况 | 返回给模型的 code |
| --- | --- |
| 参数 JSON 解析失败 | `-32700` |
| 缺少 `method` | `-32600` |
| 方法不存在（会附带可用方法列表） | `-32601` |
| `agent.InvalidParams(...)` / `Bind` 失败 / `Validate` 失败 | `-32602` |
| handler 返回普通 error 或 panic | `-32603`，message 为错误内容 |
| 用户拒绝确认 | `-32001` |
| 运行被 Stop 而未执行 | `-32002` |
| 自定义 | `agent.NewRPCError(code, msg, data)` |

所有这些错误都只回给模型，不会中断 agent loop，模型可以自行修正后重试。

### 上下文参数（身份验证）

`Config.ContextParams` 用于传递当前请求的身份信息，handler 通过 `call.Value("user_id")` 或 `call.ContextParams` 读取。它**不会**发给模型，也**不会**持久化，每次 `agent.New` 时由调用方传入当前用户的值即可——确认流程中由另一个请求来 `Confirm` 时，handler 拿到的是那次请求传入的上下文参数。

### 消息查询

```go
agent.LatestMessages(ctx, store, sid, 20)            // 最新 20 条（按时间正序）
agent.MessagesBefore(ctx, store, sid, msgID, 20)     // msgID 之前的 20 条（向上翻页）
agent.MessagesAfter(ctx, store, sid, msgID, 20)      // msgID 之后的 20 条
// Agent 上也有同名便捷方法：a.LatestMessages(ctx, 20) ...
```

`Message` 字段：`ID`、`Seq`（会话内递增序号）、`Role`、`Content`（提取出的文本）、`ToolCalls`、`ToolCallID`、`Raw`（完整原始 JSON）、`CreatedAt`。

### Token 记录与计费

每次 LLM 调用都会写一条 `agent_llm_calls` 记录，包括：`prompt_tokens`、`completion_tokens`、`total_tokens`、`cached_tokens`（缓存读）、`cache_write_tokens`、`reasoning_tokens`、音频输入/输出、预测 accepted/rejected，以及接口返回的**原始 usage JSON**、finish_reason、延迟、各项积分。兼容 OpenAI、DeepSeek（`prompt_cache_hit_tokens`）、Anthropic 风格（`cache_read_input_tokens` / `cache_creation_input_tokens`）字段。

```go
// 单一价格，单位：积分 / 百万 token
agent.Pricing{Input: 100, Output: 1000, CacheRead: 10, CacheWrite: 125} // CacheWrite 为 0 时按 Input 计

// 按模型定价
agent.PricingTable{
	Default: agent.Pricing{Input: 100, Output: 1000, CacheRead: 10},
	Models:  map[string]agent.Pricing{"gpt-4o": {Input: 250, Output: 1000, CacheRead: 125}},
}

// 或实现自己的接口
type Billing interface { Cost(model string, usage agent.Usage) agent.Cost }
```

计算方式：非缓存输入 = `prompt_tokens - cached_tokens - cache_write_tokens`，分别乘以对应单价。

查询与扣费：

```go
recs, _ := agent.ListUsage(ctx, store, sid)       // 每次调用明细
sum, _ := agent.SessionUsage(ctx, store, sid)     // 汇总
// 实时扣费：
Config.OnUsage = func(ctx context.Context, r agent.UsageRecord) { deduct(r.Cost.Total) }
```

### 前缀缓存

- 每条消息保存完整原始 JSON（`raw` 列），请求时按原字节回放，不做改写或截断。
- 系统提示词和工具定义是确定性生成的（方法按名称排序），请保持 `SystemPrompt`、`RPCDoc`、`Methods` 稳定。
- `ContextLength` 只用于估算和限制 `max_tokens`，超出时返回 `ErrContextLengthExceeded`，不会偷偷裁剪历史破坏缓存。

### 存储层

默认 provider `agent.NewSQLStore(db, dialect)` 接受 `*sql.DB`、`*sql.Tx` 或 `*sql.Conn`。SDK 的 SQL 统一用 `?` 占位符，Postgres 会自动改写为 `$n`。

也可以实现自己的 `Store`（比如接入已有的数据库网关）：

```go
type Store interface {
	Dialect() agent.Dialect
	Exec(ctx context.Context, query string, args ...any) (rowsAffected int64, err error)
	Query(ctx context.Context, query string, args ...any) (agent.Rows, error)
}
// Rows 与 *sql.Rows 一致：Next / Scan / Err / Close
```

表结构（前缀 `agent_`，时间统一为毫秒时间戳 BIGINT）：

| 表 | 内容 |
| --- | --- |
| `agent_sessions` | 会话、状态、运行锁、last_error、metadata |
| `agent_messages` | 消息，`(session_id, seq)` 唯一，`raw` 为完整原始 JSON |
| `agent_llm_calls` | 每次 LLM 调用的全部 token 字段、原始 usage、积分、延迟 |
| `agent_rpc_calls` | 每次 RPC 调用：方法、参数、状态、确认、JSON-RPC 结果 |

### 其他配置

| 字段 | 说明 |
| --- | --- |
| `ExtraBody` | 合并进请求体，如 `temperature`、`top_p`、`reasoning_effort` |
| `UseMaxCompletionTokens` | 用 `max_completion_tokens` 代替 `max_tokens`（新版 OpenAI 推理模型） |
| `Headers` / `HTTPClient` | 自定义请求头 / HTTP 客户端 |
| `MaxRetries` | 429 / 5xx / 网络错误重试次数，默认 2，负数关闭 |
| `MaxSteps` | 单次运行最多 LLM 调用次数，默认 50 |
| `StaleAfter` | 运行锁心跳超时，默认 15 分钟 |
| `OnMessage` | 每条消息落库后回调，可用于推送给前端 |
| `OnUsage` | 每次 LLM 调用记账后回调 |

## 开发

```bash
go test ./...                    # 单元测试（根模块零依赖）
cd examples && go test ./...     # 基于 SQLite + 模拟 OpenAI 服务的端到端测试
```

## License

MIT
