# agent-go

一个轻量、零第三方依赖的 Go Agent SDK，对接任意 **OpenAI 兼容**的 Chat Completions 接口。

- 模型只有**一个工具**：JSON-RPC 2.0 调用（`method` / `params` / `id`）。你注册的 Go 方法都通过它被调用，SDK 负责路由、参数错误回传、panic 兜底。
- 会话、消息、RPC 调用、每次 LLM 调用的全部 token 字段和积分都持久化到**抽象的 SQL 存储层**，通过 session id 随时恢复对话。
- 消息按原始 JSON 字节完整保存并原样回放，保证大模型**前缀缓存**不丢失。
- **流式输出**：助手消息边生成边写库（默认每 2 秒刷新一次，可配置），前端轮询数据库即可拿到实时内容。
- 支持需要人工确认的 RPC 方法、Stop / Continue、跨进程的会话锁。
- **崩溃恢复**：进程挂掉后，`Init` 会把卡住的会话、写了一半的消息、执行中的工具调用恢复成一致状态，可以直接继续对话。

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

// 入参说明写在结构体标签里：desc 说明、enum 枚举、omitempty/指针表示可选
type GetOrderParams struct {
	OrderID string `json:"order_id" desc:"订单号，如 ORD-123"`
}

// 可选：实现 Validate，校验失败会以 -32602 invalid params 返回给模型
func (p GetOrderParams) Validate() error {
	if p.OrderID == "" {
		return errors.New("order_id is required")
	}
	return nil
}

type RefundParams struct {
	OrderID string  `json:"order_id" desc:"订单号"`
	Mode    string  `json:"mode" enum:"full,partial" desc:"全额或部分退款"`
	Amount  float64 `json:"amount,omitempty" desc:"部分退款金额（元）"`
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
			// NewMethod 从 GetOrderParams 自动生成参数说明，并自动解码 + 校验 params
			agent.NewMethod("get_order", func(ctx context.Context, call *agent.Call, p GetOrderParams) (any, error) {
				userID := call.Value("user_id") // 读取当前会话的上下文参数
				return map[string]any{"order_id": p.OrderID, "owner": userID, "status": "delivered"}, nil
			}, agent.MethodDoc{
				Description: "查询当前用户的订单",
				Result:      `{"order_id": string, "status": string}`,
			}),
			agent.NewMethod("refund_order", func(ctx context.Context, call *agent.Call, p RefundParams) (any, error) {
				if p.Mode == "partial" && p.Amount <= 0 {
					return nil, agent.InvalidParams("partial refund needs a positive amount")
				}
				return map[string]any{"refunded": p.Amount}, nil
			}, agent.MethodDoc{
				Description:    "给订单退款",
				Doc:            "退款前必须先调用 get_order 确认订单金额。\n部分退款金额不能超过订单金额。",
				Examples:       []string{`{"order_id":"ORD-1","mode":"partial","amount":10}`},
				RequireConfirm: true, // 需要确认才执行
			}),
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
agent.Init(ctx, store)                                  // 建表 + 崩溃恢复，幂等，启动时调用
agent.SchemaStatements(agent.Postgres)                  // 只要 DDL，交给自己的迁移工具
sid, _ := agent.CreateSession(ctx, store, metadata)     // 创建会话，返回 session id
s, _ := agent.GetSession(ctx, store, sid)               // 会话状态、last_error、metadata
agent.ResetSession(ctx, store, sid)                     // 手动恢复单个会话（见“异常恢复”）
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
- `stopped`：调用了 `Stop`。正在流式输出的消息保留已生成部分（`interrupted`），尚未执行的调用以 `-32002 cancelled` 结果回填，保证历史合法
- `max_steps`：达到 `Config.MaxSteps`（默认 50 次 LLM 调用）

`Chat` 会阻塞到本次运行结束，Web 服务里一般放到 goroutine 中执行，前端轮询数据库获取进度（见“流式输出与前端轮询”）。`RunResult` 还包含本次新增的消息 `Messages`、`PendingCalls`、本次 `Usage` 和 `Cost`，`res.Reply()` 取最后一条助手文本。

**状态与并发**：每个会话同一时间只能有一个运行，通过数据库行锁实现（跨进程有效）。会话在运行中再调用 `Chat` 返回 `ErrBusy`；等待确认时调用 `Chat` / `Continue` 返回 `ErrWaitingConfirmation`。运行期间后台每几秒心跳一次（流式输出和长时间的 RPC 调用中也会），超过 `Config.StaleAfter`（默认 1 分钟）没有心跳的会话可被其他运行接管。进程重启时由 `Init` 统一恢复，见“异常恢复”。

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

#### 注册方法（推荐 `agent.NewMethod`）

Go 没有注解，也无法在运行时读取注释；入参说明写在**结构体标签**里，SDK 通过反射自动生成 JSON Schema：

```go
type RefundParams struct {
	OrderID string   `json:"order_id" desc:"订单号，如 ORD-123"`
	Mode    string   `json:"mode" enum:"full,partial" desc:"全额或部分退款"`
	Amount  float64  `json:"amount,omitempty" desc:"部分退款金额（元）"`
	Items   []Item   `json:"items" required:"false" desc:"部分退款的商品"`
	Note    *string  `json:"note" desc:"备注"`
}

agent.NewMethod("refund_order", handler, agent.MethodDoc{
	Description:    "给订单退款",                          // 一句话简介
	Doc:            "退款前必须先调用 get_order……\n……",      // 多行详细说明：业务规则、调用顺序
	Result:         `{"refunded": number}`,                // 返回值说明
	Examples:       []string{`{"order_id":"ORD-1","mode":"full"}`}, // params 示例（必须是合法 JSON）
	RequireConfirm: true,
})
```

| 标签 | 作用 |
| --- | --- |
| `json:"name"` | 参数名；`json:"-"` 跳过；嵌入结构体会被展开，与 `encoding/json` 一致 |
| `desc:"..."` | 参数说明 |
| `enum:"a,b,c"` | 可选值（整数 / 浮点 / 布尔字段会转成对应类型） |
| `required:"true/false"` | 覆盖默认规则。默认：非指针且没有 `omitempty` / `omitzero` 的字段为必填 |

支持 string、bool、整数、浮点、切片 / 数组、`map[string]T`、嵌套结构体、指针、`time.Time`（date-time 字符串）、`encoding.TextMarshaler`（字符串）、`any` / `json.RawMessage`（任意 JSON）。字段按声明顺序输出。

所有方法会被自动整理后拼到系统提示词末尾，最后附上 `Config.RPCDoc`（写多个方法共用的约定）。模型看到的内容类似：

```
## Methods
- `refund_order`: 给订单退款 (requires confirmation)
  params: {"type":"object","properties":{"order_id":{"type":"string","description":"订单号，如 ORD-123"},"mode":{"type":"string","description":"全额或部分退款","enum":["full","partial"]},...},"required":["order_id","mode"]}
  returns: {"refunded": number}
  example params: {"order_id":"ORD-1","mode":"full"}
  退款前必须先调用 get_order……

## Documentation
<Config.RPCDoc>
```

说明内容是确定性生成的（方法按名称排序），不会影响前缀缓存。

#### 其他写法

```go
// 1. 手写 Method：Params 可以是任意 JSON Schema，也可以用 agent.ParamsSchema[T]() 生成
agent.Method{
	Name:        "get_order",
	Description: "查询订单",
	Params:      agent.ParamsSchema[GetOrderParams](),
	Handler:     agent.Typed(getOrder), // 泛型：严格解码 params（拒绝未知字段），可选 Validate() error
}

// 2. 原始 Handler：自己解码 / 校验
Handler: func(ctx context.Context, call *agent.Call) (any, error) {
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
| 运行被 Stop / 进程重启而未执行 | `-32002` |
| 执行过程中进程崩溃（可能已生效，也可能没有） | `-32003` |
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

`Message` 字段：`ID`、`Seq`（会话内递增序号）、`Role`、`Status`、`Content`（提取出的文本）、`Reasoning`（思考过程文本）、`ToolCalls`、`ToolCallID`、`Raw`（完整原始 JSON）、`CreatedAt`、`UpdatedAt`。另有 `agent.GetMessage(ctx, store, sid, msgID)` 查单条。

### 流式输出与前端轮询

SDK 始终以流式（SSE）调用大模型。前端不需要连接 SDK，只要**轮询数据库**：

- 助手消息在收到第一个片段时就插入数据库（`status = streaming`），之后每隔 `Config.StreamFlushInterval`（默认 2 秒）更新一次内容，流结束时再写最终版本（`done`）。
- 模型一返回工具调用，就为每个调用插入一条 tool 消息（`pending`）；开始执行前改成 `running`，执行完写入结果（`done`）。顺序和模型给出的 `tool_calls` 一致。

| `Message.Status` | 含义 | 还会变吗 |
| --- | --- | --- |
| `streaming` | 助手消息生成中 | 会 |
| `pending` | 工具调用排队中 / 等待确认 | 会 |
| `running` | 工具调用执行中 | 会 |
| `done` | 已完成 | 不会 |
| `interrupted` | 助手输出被 Stop / 报错 / 崩溃打断，保留已生成的部分 | 不会 |

用 `m.Status.Final()` 判断消息是否定稿。推荐的前端轮询方式：**游标只推进到最后一条已定稿的消息**，这样还在变化的消息每次都会被重新拉到：

```go
cursor := "" // 最后一条已定稿消息的 ID
for {
	var ms []agent.Message
	if cursor == "" {
		ms, _ = agent.LatestMessages(ctx, store, sid, 5)
	} else {
		ms, _ = agent.MessagesAfter(ctx, store, sid, cursor, 50)
	}
	render(ms) // 按 ID 覆盖渲染
	for _, m := range ms {
		if !m.Status.Final() {
			break
		}
		cursor = m.ID
	}
	s, _ := agent.GetSession(ctx, store, sid)
	if s.Status != agent.StatusRunning { /* 空闲或等待确认 */ }
	time.Sleep(time.Second)
}
```

同一进程内也可以用 `Config.OnStream`（每个文本片段）和 `Config.OnMessage`（每次写库）回调。

### 异常恢复

`agent.Init` 启动时会恢复上次进程留下的会话（状态为 `running` 的会话）：

| 崩溃时所处阶段 | 恢复后 |
| --- | --- |
| 会话状态 `running` | 改为 `idle`；如果还有等待确认的调用则为 `waiting_confirmation`。`last_error` 记录恢复原因 |
| 大模型流式输出中 | 助手消息改为 `interrupted`，保留已写入的部分内容（最多丢失最后一个刷新间隔的内容），可以直接继续对话 |
| 工具调用执行中（`running`） | 调用标记为 `failed`，tool 消息写入 `-32003` 错误：“系统崩溃，调用可能已生效也可能没有” |
| 工具调用已排队但还没开始 | 标记为 `cancelled`，写入 `-32002` 错误 |
| 等待确认的调用 | 保留，恢复后仍可 `Confirm` |

恢复后历史是完整合法的，直接 `Chat` / `Continue` 即可，模型能看到哪些调用失败了。被打断的助手消息如果一个字都没有，不会发给模型。

**多实例部署**：默认 `Init` 会恢复**所有** `running` 的会话，适合单实例。多个进程共用一个数据库时，要只恢复真正没有心跳的会话，避免重启一个实例时误伤其他实例正在跑的会话：

```go
agent.Init(ctx, store, agent.RecoverStaleAfter(2*time.Minute)) // 需大于 Config.StaleAfter
```

另外，每次运行开始时也会自动修复该会话的残留状态（接管无心跳的会话时同样适用），所以即使没有重新调用 `Init`，也不会出现历史不完整的情况。

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
| `agent_messages` | 消息，`(session_id, seq)` 唯一，`raw` 为完整原始 JSON，`status` 为流式 / 执行状态 |
| `agent_llm_calls` | 每次 LLM 调用的全部 token 字段、原始 usage、积分、延迟 |
| `agent_rpc_calls` | 每次 RPC 调用：方法、参数、状态、确认、JSON-RPC 结果 |

### 其他配置

| 字段 | 说明 |
| --- | --- |
| `StreamFlushInterval` | 流式输出时写库的间隔，默认 2 秒 |
| `StreamIdleTimeout` | 流式输出多久没有数据就中断，默认 2 分钟，负数关闭 |
| `KeepReasoning` | 把思考内容以 `reasoning_content` 回传给模型（DeepSeek 思考模式 + 工具调用需要）。无论开不开，思考内容都会存进 `Message.Reasoning` |
| `ExtraBody` | 合并进请求体，如 `temperature`、`top_p`、`reasoning_effort`；值为 `nil` 表示删除默认字段，例如不支持 `stream_options` 的服务可设 `"stream_options": nil` |
| `UseMaxCompletionTokens` | 用 `max_completion_tokens` 代替 `max_tokens`（新版 OpenAI 推理模型） |
| `Headers` / `HTTPClient` | 自定义请求头 / HTTP 客户端（默认不设整体超时，由 `StreamIdleTimeout` 兜底） |
| `MaxRetries` | 429 / 5xx / 网络错误重试次数，默认 2，负数关闭；已经开始输出后不再重试 |
| `MaxSteps` | 单次运行最多 LLM 调用次数，默认 50 |
| `StaleAfter` | 运行锁心跳超时，默认 1 分钟 |
| `OnStream` | 每个流式文本片段的回调 |
| `OnMessage` | 每次消息写库（插入、流式刷新、工具状态变化）后回调 |
| `OnUsage` | 每次 LLM 调用记账后回调 |

## 开发

```bash
go test ./...                    # 单元测试（根模块零依赖）
cd examples && go test ./...     # 基于 SQLite + 模拟 OpenAI 服务的端到端测试
```

## License

MIT
