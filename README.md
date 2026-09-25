# agent-go

一个轻量、零第三方依赖的 Go Agent SDK，对接任意 **OpenAI 兼容**的 Chat Completions 接口。

- 模型的核心工具是 **JSON-RPC 2.0 调用**（`method` / `params` / `id`）。你注册的 Go 方法都通过它被调用，SDK 负责路由、参数错误回传、panic 兜底。另有可选的内置工具 `view_image`，让模型通过 URL 查看图片。
- 会话、消息、RPC 调用、每次 LLM 调用的全部 token 字段和积分都持久化到**抽象的 SQL 存储层**，通过 session id 随时恢复对话。
- 消息按原始 JSON 字节完整保存并原样回放，保证大模型**前缀缓存**不丢失。
- **流式输出**：助手消息边生成边写库（默认每 2 秒刷新一次，可配置），前端轮询数据库即可拿到实时内容。
- 支持需要人工确认的 RPC 方法、Stop / Continue、跨进程的会话锁。
- **文档挂载**：挂载一个 markdown 文档目录，文档目录（frontmatter）进系统提示词，模型用 `read_doc` 按需查阅；系统提示词也可以从文件加载。
- **上下文压缩**（默认“固定上个用户消息起点”，可选摘要模式）、运行中**追加用户消息队列**、每条用户消息附加 **reminder**、RPC 结果**长度保护**。
- **崩溃恢复**：进程挂掉后，`NewClient` 启动时会把卡住的会话、写了一半的消息、执行中的工具调用恢复成一致状态，可以直接继续对话。

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

	// 1. 启动时创建一个 Client：存储层、计费、大模型和 RPC 方法等全局配置只设置这一次。
	//    它会建表（幂等），并恢复上次进程崩溃留下的会话。
	db, _ := sql.Open("sqlite", "file:agent.db?_pragma=busy_timeout(5000)")
	client, err := agent.NewClient(ctx, agent.Config{
		Store:           agent.NewSQLStore(db, agent.SQLite),
		Billing:         agent.Pricing{Input: 100, Output: 1000, CacheRead: 10}, // 每百万 token 的积分
		BaseURL:         "https://api.openai.com/v1", // 任意 OpenAI 兼容地址
		APIKey:          "sk-...",
		Model:           "gpt-4o-mini",
		ContextLength:   128000, // 模型上下文长度
		MaxOutputTokens: 4096,   // 最大输出
		SystemPrompt:    "你是一个客服助手。",
		RPCDoc:          "订单号格式为 ORD-123，金额单位为元。", // JSON-RPC 文档
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

	// 2. 创建会话，拿到 session id（可附带 metadata，比如所属用户）
	sessionID, _ := client.CreateSession(ctx, map[string]any{"user_id": "u_42"})

	// 3. 为这个会话创建 agent，只传和本次请求有关的参数
	a, err := client.Agent(ctx, sessionID, agent.AgentOptions{
		ContextParams: map[string]any{"user_id": "u_42", "token": "..."}, // 身份等上下文，不会发给模型
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

### Client：全局初始化与会话

所有操作都从 `*agent.Client` 发起。它在启动时创建一次，全进程共享，并保存 `Store` 等全局配置，所以后续方法都不需要再传存储层。每个方法第一个参数是 `ctx`，这是 Go 的惯例：它代表这一次调用，用于取消和超时，不能存起来复用。

```go
client, err := agent.NewClient(ctx, agent.Config{...}) // 建表 + 崩溃恢复，幂等
agent.SchemaStatements(agent.Postgres)                 // 只要 DDL，交给自己的迁移工具（配合 Config.SkipSchema）

sid, _ := client.CreateSession(ctx, metadata)          // 创建会话，返回 session id
s, _ := client.Session(ctx, sid)                       // 会话状态、last_error、metadata
client.Stop(ctx, sid)                                  // 不创建 agent 也能停止会话（比如“停止”按钮的接口）
client.PendingCalls(ctx, sid)                          // 待确认的 RPC 调用
client.Enqueue(ctx, sid, "补充一点……")                 // 运行中追加用户消息（见“消息队列”）
client.ResetSession(ctx, sid)                          // 手动恢复单个会话（见“异常恢复”）
client.DeleteSession(ctx, sid)                         // 删除会话（见下文）
client.Recover(ctx)                                    // 重新执行一次启动时的崩溃恢复
```

**删除会话**：`client.DeleteSession` 会永久删除会话本身、所有消息、RPC 调用记录和队列消息。token 用量和计费记录（`agent_llm_calls`）会**保留**：它们只包含 token 数和积分，不含对话内容，保留下来是为了让未结算的用量仍然可以结算。运行中或停止中的会话会返回 `ErrBusy`，需要先 `Stop`；会话不存在时返回 `ErrSessionNotFound`。SDK 不提供按用户查询会话的功能，session id 由调用方自己保存。

`Config` 里的大模型配置（`BaseURL`、`Model` 等）只在创建 agent 时需要。只用来查消息、做结算的服务，传一个 `Store` 就够了。

### 创建 Agent

```go
a, err := client.Agent(ctx, sid, agent.AgentOptions{
	ContextParams: map[string]any{"user_id": uid}, // 本次请求的身份，传给 RPC handler
	Override: func(cfg *agent.Config) {            // 可选：只对这个 agent 修改配置
		cfg.SystemPrompt = "你是退款专员。"
		cfg.Model = "gpt-4o"
	},
})
```

`sid` 为空时会自动创建新会话，用 `a.SessionID()` 取回。创建 agent 很轻量，每个请求新建一个即可。

### Agent 执行方法

| 方法 | 说明 |
| --- | --- |
| `a.Chat(ctx, prompt)` | 追加用户消息并运行 agent loop，阻塞直到结束 |
| `a.ChatMessage(ctx, rawJSON)` | 同上，自己构造用户消息（多模态图片等） |
| `a.Continue(ctx)` | 不加新消息继续运行（Stop 后、报错后、输出被截断后） |
| `a.Stop(ctx)` | 停止会话，**阻塞到会话变为 `idle`** 才返回；幂等，可并发、可跨进程调用（详见下文“Stop”） |
| `a.Confirm(ctx, decisions...)` | 批准 / 拒绝待确认的 RPC 调用，全部处理完后继续 loop |
| `a.Status(ctx)` | `idle` / `running` / `stopping` / `waiting_confirmation` |
| `a.PendingCalls(ctx)` | 待确认的 RPC 调用列表 |
| `a.Send(ctx, prompt)` | **推荐前端统一使用**：会话空闲就直接开始对话，运行中就交给当前这次运行，不用自己判断状态（见“消息队列”） |
| `a.Enqueue(ctx, prompt)` | 只放进队列，下一次调用大模型前插入（见“消息队列”） |
| `a.Session(ctx)` / `a.SessionID()` / `a.Client()` | 会话信息 / 会话 ID / 所属 Client |

一次运行（`Chat` / `Continue` / `Confirm`）会在以下情况返回，`RunResult.StopReason` 说明原因：

- `completed`：模型给出不含工具调用的回复
- `waiting_confirmation`：有 `RequireConfirm` 的方法等待确认（同一轮中不需要确认的调用已先执行）
- `stopped`：调用了 `Stop`，见下文
- `max_steps`：达到 `Config.MaxSteps`（默认 50 次 LLM 调用）

`Chat` 会阻塞到本次运行结束，Web 服务里一般放到 goroutine 中执行，前端轮询数据库获取进度（见“流式输出与前端轮询”）。`RunResult` 还包含本次新增的消息 `Messages`、`PendingCalls`、本次 `Usage` 和 `Cost`，`res.Reply()` 取最后一条助手文本。

**状态与并发**：每个会话同一时间只能有一个运行，通过数据库行锁实现（跨进程有效）。会话在运行中再调用 `Chat` 返回 `ErrBusy`；等待确认时调用 `Chat` / `Continue` 返回 `ErrWaitingConfirmation`。运行期间后台每几秒心跳一次（流式输出和长时间的 RPC 调用中也会），超过 `Config.StaleAfter`（默认 1 分钟）没有心跳的会话可被其他运行接管。进程重启时由 `NewClient` 统一恢复，见“异常恢复”。

典型 Web 服务用法：一个请求里 `go a.Chat(...)`，另一个请求用同一个 session id `client.Agent(...)` 后调 `Confirm`，或者直接 `client.Stop(ctx, sid)` / `client.Session(ctx, sid)` / `client.PendingCalls(ctx, sid)`。

### Stop

状态流转：

```
idle ──Chat/Continue──▶ running ──完成──▶ idle
                          │    └──需要确认──▶ waiting_confirmation ──Confirm──▶ running
                          │                          │
                          └──Stop──▶ stopping ──▶ idle ◀──Stop──┘
```

调用 `Stop` 后：

- **大模型输出中**：立刻取消请求，已生成的文本保留为不完整的助手消息（`interrupted`），会进入后续对话历史。
- **工具调用执行中**：**不等待** handler，立刻把结果写成 `stopped by user while this call was running; it may or may not have taken effect`。handler 的 ctx 会被取消，它之后返回的结果会被丢弃，不会写入数据库（Go 无法强杀 goroutine，handler 应该尊重 ctx）。
- **未执行的调用**（排队中、已批准未执行、**等待确认中**）：结果写成 `stopped by user: this call was not executed`。
- 以上错误码都是 `-32002`，全部写完后会话变为 `idle`。**Stop 之后会话总是 `idle`**，不会停在 `waiting_confirmation`。

并发与竞争：

| 情况 | 行为 |
| --- | --- |
| 重复 / 并发调用 `Stop` | 幂等。都会等到 `idle` 后返回 `nil` |
| 会话已是 `idle` | 立即返回 `nil` |
| 会话是 `waiting_confirmation` | 取消所有待确认调用，变为 `idle` |
| `Stop` 与运行自然结束同时发生 | 如果运行结束时发现状态已是 `stopping`，按停止处理；如果运行抢先变成 `waiting_confirmation`，`Stop` 会接着取消待确认调用。最终都是 `idle` |
| `Stop` 返回后立刻 `Chat` | 可以，不会 `ErrBusy` |
| `stopping` 期间调用 `Chat` | 返回 `ErrBusy` |
| 另一个进程调用 `Stop` | 运行方每隔 `StopPollInterval`（默认 1 秒）读一次会话状态，约 1 秒内停止 |
| 持有会话的进程已经挂了 | 超过 `StaleAfter` 没有心跳时，`Stop` 自己接管并清理（等同崩溃恢复），不会一直等 |
| 传给 `Stop` 的 ctx 超时 | 返回 ctx 的错误，但停止流程仍会继续完成 |

`Chat` 被停止时返回 `RunResult{StopReason: "stopped", Status: "idle"}`，`err` 为 `nil`。

### RPC 方法

模型调用你的方法用的工具默认叫 `json_rpc`（`Config.ToolName` 可改），参数是一个 JSON-RPC 请求：

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
| handler 返回普通 error 或 panic | `-32603`。message 为错误内容；panic 时只给模型 panic 的值，完整堆栈写入日志 |
| 用户拒绝确认 | `-32001` |
| 超过超时时间（`Method.Timeout` / `Config.RPCTimeout`） | `-32004` |
| 被 Stop（执行中或未执行）/ 进程重启时未执行 | `-32002` |
| 执行过程中进程崩溃（可能已生效，也可能没有） | `-32003` |
| 自定义 | `agent.NewRPCError(code, msg, data)` |

所有这些错误都只回给模型，不会中断 agent loop，模型可以自行修正后重试。

### 文档挂载（read_doc）与系统提示词文件

挂载一个 markdown 文档目录，模型会按需查阅：

```go
//go:embed docs
var docsFS embed.FS

sub, _ := fs.Sub(docsFS, "docs")
client, _ := agent.NewClient(ctx, agent.Config{
	// ...
	Docs:             sub,                 // 或 os.DirFS("./docs")
	SystemPromptFile: "prompts/system.md", // 核心系统提示词放在文件里，而不是写在代码中
	DocPageChars:     20000,               // 长文档分页，默认 20000 字符一页
})
client.Docs() // 已挂载的文档列表
```

每个文档开头是 frontmatter，`title` 必填且**不能重复**（不区分大小写），其他字段可选：

```markdown
---
title: Refund policy
description: 退款时限、部分退款和例外情况
audience: customers
---
# 退款
……
```

- 递归扫描目录下所有 `.md` / `.markdown` 文件，在 `NewClient` 时加载一次。缺少 title、title 重复或文件读不到时，`NewClient` 直接报错（fail fast）。
- 所有文档的 frontmatter（title、description 和其他字段）按标题排序后写进系统提示词。**正文不会放进上下文**，模型需要时调用内置工具 `read_doc`，参数为 `{"title": "...", "page": 2}`。标题匹配不区分大小写，标题不存在时会返回错误，提示模型从列表里选。
- `read_doc` 返回纯文本（`# 标题` + 正文，去掉 frontmatter）。超过 `DocPageChars` 的文档会优先在段落处分页，结果里会注明第几页、共几页，以及怎么读下一页。
- `SystemPromptFile`：设置了 `Docs` 时是 Docs 里的相对路径，否则是本地文件路径。frontmatter 会被去掉，这个文件本身不会出现在文档列表中。同时设置了 `SystemPrompt` 字符串时，追加在文件内容之后。
- 系统提示词的顺序：提示词文件 → `SystemPrompt` → 文档列表 → RPC 方法说明，整体稳定不变，不影响前缀缓存。修改文档后需要重新创建 Client 才会生效。

实测：挂载 5 个文档，模型能为每个问题选对文档；一本 4 页的手册，它会逐页翻到最后一页找到答案；文档里没有的问题，它会明确说“文档中没有”。

### 查看图片（view_image）

开启 `Config.ViewImage` 后，除了 `json_rpc`，模型还会多一个内置工具 `view_image`，参数只有一个 `url`。适合让模型查看用户发来的图片链接，或者某个 RPC 方法返回的图片地址。

```go
client, _ := agent.NewClient(ctx, agent.Config{
	// ...
	ViewImage:       true,  // 仅对支持视觉的模型开启，纯文本模型收到 image_url 会报错
	ViewImageDetail: "low", // 可选：low / high / auto，low 消耗的图片 token 少很多
})
```

SDK **不下载图片，也不转 Base64**，只把 URL 原样交给模型，由大模型服务商自己去拉取（所以 URL 必须是公网可访问的）。OpenAI 兼容接口的 tool 消息只能放文本，所以图片会作为紧跟在后面的一条独立 user 消息发给模型：

```
assistant  tool_calls: [view_image {"url": "https://cdn.example.com/cat.png"}]
tool       {"ok":true,"url":"https://cdn.example.com/cat.png","note":"The image is attached in the next user message."}
user       [{"type":"text","text":"[view_image result for call_x] https://cdn.example.com/cat.png"},
            {"type":"image_url","image_url":{"url":"https://cdn.example.com/cat.png","detail":"low"}}]
```

- 同一轮有多个工具调用时，图片消息放在**这一轮所有 tool 消息之后**，保证工具调用和结果的配对不被打断。
- 这条 user 消息的 `Message.Kind` 为 `"view_image"`，`ToolCallID` 指向对应的工具调用。前端可以据此把它渲染成附件，而不是用户说的话。
- 只接受 `http` / `https` 的 URL。其他地址（`file://`、`data:` 等）会以 `{"ok":false,"error":...}` 返回给模型，不会生成图片消息。
- 进程崩溃时如果图片消息还没写入，下次运行开始时会自动补上，不会重复。
- **图片加载失败不会卡死会话**：服务商拉取不到图片时（URL 404、防盗链等），会以 400 拒绝整个请求。如果不处理，这条图片消息每次都会被重放，会话就永久失败了。SDK 检测到这种情况时，会把本轮还没成功发送的图片消息标记为 `excluded`（前端仍可展示，但不再发给模型），把对应的工具结果改写为 `{"ok":false,"error":"...could not load this image..."}`，然后自动重试。模型会知道图片打不开，对话照常继续。

#### 并行执行

模型在一轮里发出多个调用时，默认**并行执行**。`Config.ToolConcurrency` 用来限制同时执行的数量：`0`（默认）表示全部同时执行，`1` 表示逐个执行，`n` 表示最多同时执行 n 个。每个调用的 tool 消息在执行前就已经按模型给出的顺序建好，结果写回原位，所以模型看到的顺序与完成先后无关。并行执行时，handler 和 `OnMessage` 回调需要是并发安全的。Stop 和超时对每个调用分别生效。

实测：模型一轮发出 4 个查询，每个耗时 2 秒，并行执行后工具阶段总共 2.0 秒。

#### 超时

默认**不设超时**。需要时可以设置全局的 `Config.RPCTimeout`，或者单独设置某个方法的 `Method.Timeout` / `MethodDoc.Timeout`。超时后 handler 的 ctx 会被取消，模型立即收到 `-32004` 错误（“timed out after 5s; the call may or may not have taken effect”），不会等待 handler 返回。不响应 ctx 的 handler 会在后台继续执行，但它的返回结果会被丢弃。

### 上下文参数（身份验证）

`AgentOptions.ContextParams` 用于传递当前请求的身份信息，handler 通过 `call.Value("user_id")` 或 `call.ContextParams` 读取。它**不会**发给模型，也**不会**持久化，每次 `client.Agent` 时由调用方传入当前用户的值即可——确认流程中由另一个请求来 `Confirm` 时，handler 拿到的是那次请求传入的上下文参数。

### 消息查询

```go
client.LatestMessages(ctx, sid, 20)           // 最新 20 条（按时间正序）
client.MessagesBefore(ctx, sid, msgID, 20)    // msgID 之前的 20 条（向上翻页）
client.MessagesAfter(ctx, sid, msgID, 20)     // msgID 之后的 20 条
client.Message(ctx, sid, msgID)               // 单条
```

`Message` 字段：`ID`、`Seq`（会话内递增序号）、`Role`、`Kind`（普通消息为空；`view_image` 注入的图片消息为 `"view_image"`；压缩点为 `"compaction"`）、`RefSeq`（压缩点之后模型从哪条消息开始看）、`Status`、`Content`（提取出的文本）、`Reasoning`（思考过程文本）、`ToolCalls`、`ToolCallID`、`Raw`（完整原始 JSON）、`CreatedAt`、`UpdatedAt`。

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
| `excluded` | 保留用于展示，但不再发给模型（例如服务商无法加载的 `view_image` 图片） | 不会 |

用 `m.Status.Final()` 判断消息是否定稿。推荐的前端轮询方式：**游标只推进到最后一条已定稿的消息**，这样还在变化的消息每次都会被重新拉到：

```go
cursor := "" // 最后一条已定稿消息的 ID
for {
	var ms []agent.Message
	if cursor == "" {
		ms, _ = client.LatestMessages(ctx, sid, 5)
	} else {
		ms, _ = client.MessagesAfter(ctx, sid, cursor, 50)
	}
	render(ms) // 按 ID 覆盖渲染
	for _, m := range ms {
		if !m.Status.Final() {
			break
		}
		cursor = m.ID
	}
	s, _ := client.Session(ctx, sid)
	if s.Status != agent.StatusRunning { /* 空闲或等待确认 */ }
	time.Sleep(time.Second)
}
```

同一进程内也可以用 `Config.OnStream`（每个文本片段）和 `Config.OnMessage`（每次写库）回调。

### 异常恢复

`agent.NewClient` 启动时会恢复上次进程留下的会话（状态为 `running` / `stopping` 的会话）：

| 崩溃时所处阶段 | 恢复后 |
| --- | --- |
| 会话状态 `running` | 改为 `idle`；如果还有等待确认的调用则为 `waiting_confirmation`。`last_error` 记录恢复原因 |
| 会话状态 `stopping` | 按停止处理完（包括取消待确认调用），改为 `idle` |
| 大模型流式输出中 | 助手消息改为 `interrupted`，保留已写入的部分内容（最多丢失最后一个刷新间隔的内容），可以直接继续对话 |
| 工具调用执行中（`running`） | 调用标记为 `failed`，tool 消息写入 `-32003` 错误：“系统崩溃，调用可能已生效也可能没有” |
| 工具调用已排队但还没开始 | 标记为 `cancelled`，写入 `-32002` 错误 |
| 等待确认的调用 | 保留，恢复后仍可 `Confirm` |

恢复后历史是完整合法的，直接 `Chat` / `Continue` 即可，模型能看到哪些调用失败了。被打断的助手消息如果一个字都没有，不会发给模型。

**多实例部署**：默认会恢复**所有** `running` 的会话，适合单实例。多个进程共用一个数据库时，要只恢复真正没有心跳的会话，避免重启一个实例时误伤其他实例正在跑的会话：

```go
agent.NewClient(ctx, agent.Config{
	Store:             store,
	RecoverStaleAfter: 2 * time.Minute, // 需大于 Config.StaleAfter
	// ...
})
```

另外，每次运行开始时也会自动修复该会话的残留状态（接管无心跳的会话时同样适用），所以即使进程没有重启，也不会出现历史不完整的情况。

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

查询：

```go
recs, _ := client.ListUsage(ctx, sid)   // 每次调用明细
sum, _ := client.Usage(ctx, sid)        // 累计汇总（含已结算）
```

#### 结算（扣积分）

积分只在每次 LLM 调用结束时产生，每条 `agent_llm_calls` 记录都带结算状态：未结算、已认领（`BillID` 有值、`BilledAt` 为空）、已结算（`BilledAt` 有值）。

**推荐：`SettleUsage` 一步结算**，并发安全，不会重复扣费：

```go
bill, err := client.SettleUsage(ctx, sid, func(ctx context.Context, b *agent.Bill) error {
	// b.Summary.Cost.Total 为本次要扣的积分，b.Records 为明细
	// 用 b.ID 作为幂等键写入你的账本
	return wallet.Deduct(ctx, userID, b.Summary.Cost.Total, b.ID)
})
// bill == nil && err == nil 表示没有待结算的记录
```

流程：先用一条原子 `UPDATE` 把该会话所有未结算记录认领到新的 `bill.ID` 下（同一条记录只能被认领一次，多个 worker 并发结算也不会重复），然后调用你的 `charge`：

- `charge` 返回 `nil` → 记录标记为已结算；
- `charge` 返回错误（如余额不足）→ 记录释放回未结算，下次再结算，错误原样返回；
- 进程在 `charge` 期间崩溃 → 记录保持“已认领”，在 `UnbilledUsage` 里能看到 `BillID`。去账本查这个 ID，已扣则 `client.CompleteBill(ctx, billID)`，未扣则 `client.ReleaseBill(ctx, billID)`。

**也可以手动查询 + 标记**：

```go
sum, recs, _ := client.UnbilledUsage(ctx, sid)  // 未结算的汇总和明细；sid 传 "" 表示所有会话
// ... 自己扣费 ...
client.MarkBilled(ctx, "你的账单号", recordIDs...)  // 已结算的记录不会被重复标记
```

需要实时处理的话，还可以用 `Config.OnUsage` 回调，每次 LLM 调用记账后触发。

#### 预算控制（BeforeLLMCall）

结算是事后的。为了防止一次长的工具调用循环把余额扣成负数，可以在**每次调用大模型之前**（包括压缩时的摘要调用）检查余额：

```go
agent.Config{
	BeforeLLMCall: func(ctx context.Context, info agent.LLMCallInfo) error {
		// info: SessionID、Model、Purpose（"chat" 或 "summary"）、EstimatedPromptTokens、MaxTokens，
		// 以及 MaxCost：按计费规则计算的最坏情况花费（整个 prompt 按未缓存输入计，加上 MaxTokens 的输出）
		if balance(ctx) < info.MaxCost.Total {
			return ErrInsufficientCredits
		}
		return nil
	},
}
```

返回错误时这次请求不会发出，也不会产生任何花费。运行会结束，`Chat` 原样返回这个错误（可以用 `errors.Is` 判断），会话回到 `idle`，并把原因记在 `last_error` 里。

实测：预算 1.0 积分、让模型无限循环调用工具，跑了 8 次调用、花费 0.66 后被拦下，没有超支。

注意：被 Stop 打断或出错的流式请求，服务商通常不会返回 usage，这部分消耗无法记录。

### 上下文压缩

设置了 `ContextLength` 后，每次调用大模型前都会估算这次请求的大小。估算方法是上一次调用的真实 token 数，加上之后新增消息的字节数除以 3。超过 `CompactionThreshold`（默认 80%）时会先压缩。压缩会插入一条 `Kind = "compaction"` 的消息，**之后发给模型的历史从这条压缩消息开始**。从 session id 恢复对话时也一样；没有压缩消息就从第一条开始。数据库里的消息不会被删除，前端照常可以看到完整历史。

| `Config.Compaction` | 行为 |
| --- | --- |
| `agent.CompactionAnchor`（默认） | **固定上个用户消息起点**：把起点移到最近一条用户消息，之前的消息不再发给模型。压缩消息只是给前端看的分隔标记（`status = excluded`） |
| `agent.CompactionSummary` | 先让模型把要丢弃的部分总结成笔记（这次调用照常计费），笔记以 `<conversation_summary>` 用户消息的形式放在最前面，后面接最近一条用户消息及之后的内容 |
| `agent.CompactionOff` | 不压缩，放不下时返回 `ErrContextLengthExceeded` |

```go
agent.Config{
	ContextLength:       100000,
	Compaction:          agent.CompactionSummary, // 默认 CompactionAnchor
	CompactionThreshold: 0.8,
	CompactionPrompt:    "……",                    // 可选：自定义摘要要求
}
```

- **起点不是固定条数**：起点只在压缩的那一刻移动，两次压缩之间发送的历史开头完全不变，**前缀缓存持续有效**。实测一次对话中，每轮都命中了上一轮几乎全部的 prompt。
- 起点总是用户消息，工具调用和结果的配对不会被截断。
- 如果当前这一轮本身（从最近一条用户消息开始）已经太大，就没有可以丢弃的内容，会继续执行直到放不下（`ErrContextLengthExceeded`）。运行中用“消息队列”追加的用户消息也可以作为新的起点。
- 摘要模式下，交给模型总结的对话会按 token 预算截取：每条长消息先单独截短，仍然超长时去掉中间部分，保留开头（上一次的摘要、最早的关键信息）和结尾。

实测（`ContextLength: 5000`，每轮约 600 token 的长回答）：prompt 从 62 增长到 3786 token 后触发压缩，下一次请求降到 37 token（anchor 模式）或 224 token（summary 模式，模型依然记得最开始告诉它的暗号）。

### 消息队列（运行中追加用户消息）

**推荐用 `a.Send`**：前端发消息时不用关心会话当前在做什么：

```go
r, err := a.Send(ctx, "另外，其中一位是素食者")
// r.Run != nil：会话原本空闲，Send 自己开始了一次运行，r.Run 是这次运行的结果（和 Chat 一样会阻塞到结束）
// r.Queued：消息交给了正在进行的运行（由它来回答），或者排在待确认调用之后（确认后发出）
```

1. 消息先存进队列，进程崩溃也不会丢。
2. 会话空闲时，Send 自己开始一次运行来处理这条消息。
3. 会话正在运行时，等待当前运行在下一次调用大模型前取走这条消息，然后返回 `Queued`。**如果当前运行恰好在取走之前就结束了**（消息正好在运行结束的那一刻到达），Send 会自己再开始一次运行。所以通过 Send 发出的消息一定会得到回答。
4. 会话在等待确认时，立即返回 `Queued`，消息会在 `Confirm` 之后发出。

也可以直接使用底层的队列接口：

```go
id, _ := a.Enqueue(ctx, "另外，其中一位是素食者")        // 或 client.Enqueue(ctx, sid, ...)
client.QueuedMessages(ctx, sid)                          // 还在队列里、没发出去的消息
client.CancelQueued(ctx, sid, id)                        // 发出去之前可以撤回
client.EnqueueMessage(ctx, sid, rawJSON)                 // 多模态消息
```

- 队列存在数据库里（`agent_queued_messages`）。**下一次调用大模型之前**，队列中的所有消息按顺序各自作为一条 `role=user` 消息插入对话，同时从队列中删除，然后一次性发给模型。
- 如果模型已经给出最终回答，队列里却还有消息，agent 会继续下一轮，回应这些补充内容，不会把它们漏掉。
- 会话空闲时入队的消息，会在下一次 `Chat` / `Continue` 开始时插入，排在新问题之前。
- 插入时使用确定性的消息 ID（`msg_<队列ID>`），进程崩溃也不会重复插入。

### Reminder

```go
agent.Config{Reminder: "回答要简洁，金额一律用人民币。"}
```

每条用户消息（`Chat`、`ChatMessage`、队列消息）发给模型时，末尾都会附加 `<reminder>…</reminder>`。多模态消息则追加一个文本片段。附加后的内容会写进 `raw`，以后重放时字节不变，不影响前缀缓存；`Message.Content` 里保存的仍是用户输入的原文，前端不会显示 reminder。

### 前缀缓存

- OpenAI、DeepSeek、Gemini 等服务商会**自动**缓存相同的前缀（通常要求 prompt 超过 1024 token），不需要额外设置。
- **Claude 模型**（Anthropic 接口，或通过 OpenRouter 等网关调用）需要显式标记缓存断点。开启 `Config.CacheControl` 后，每次请求会在系统提示词和最后一条消息上加 `cache_control: {"type":"ephemeral"}`，最后一条消息上的断点随对话往后移动。这个标记只加在发出去的请求副本上，数据库里的消息不会被修改。实测对 OpenAI、Gemini 模型开启也不会报错；Claude 上的缓存效果还没有实测过（当前代理没有配置 Claude 模型）。
- 每条消息保存完整原始 JSON（`raw` 列），请求时按原字节回放，不做改写或截断。
- 系统提示词和工具定义是确定性生成的（方法按名称排序），请保持 `SystemPrompt`、`RPCDoc`、`Methods` 稳定。
- 历史只在压缩时才会改变起点（见“上下文压缩”），两次压缩之间发给模型的前缀完全不变。

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
| `agent_llm_calls` | 每次 LLM 调用的全部 token 字段、原始 usage、积分、延迟、结算状态（`bill_id` / `claimed_at` / `billed_at`） |
| `agent_queued_messages` | 运行中追加、还没发给模型的用户消息 |
| `agent_rpc_calls` | 每次 RPC 调用：方法、参数、状态、确认、JSON-RPC 结果 |

### 日志

SDK 使用标准库的 `log/slog`：

```go
agent.Config{
	Logger:   slog.New(slog.NewJSONHandler(os.Stdout, nil)), // 默认 slog.Default()
	LogLevel: slog.LevelDebug,                               // 默认 Info
}
```

- **Info**：运行开始和结束（状态、停止原因、耗时、token 数、积分）、上下文压缩、Stop、崩溃恢复、大模型请求重试和失败、RPC 超时、`BeforeLLMCall` 拒绝。
- **Debug**：每次大模型请求（消息数、估算的 token、max_tokens）和响应（finish_reason、token、延迟）、每次 RPC 调用（方法、耗时、状态）、队列消息插入。
- **Error**：RPC handler panic，附带完整堆栈。

`LogLevel` 按 agent 生效（可以通过 `AgentOptions.Override` 单独调整），所以多个 agent 可以共用同一个 logger，各自使用不同的级别。每条日志都带 `component=agent-go` 和 `session`。

### 性能说明

- 每次调用大模型前，只读取**最近一次压缩点之后**的消息（两次带索引的查询），开销不会随会话总长度增长。
- 运行中的会话会在后台每隔 `StopPollInterval`（默认 1 秒）读一次自己的会话行，用来发现其他进程发来的 Stop，另外每隔几秒更新一次心跳时间。同一进程内的 Stop 直接在内存中取消，不依赖轮询。单进程部署可以设为负数关闭轮询：其他进程发来的 Stop 会在下一步开始前生效。

### 其他配置

| 字段 | 说明 |
| --- | --- |
| `ViewImage` / `ViewImageDetail` | 开启内置的 `view_image` 工具 / 设置 `image_url.detail`，见“查看图片” |
| `Docs` / `SystemPromptFile` / `DocPageChars` | 挂载文档目录 / 从文件加载系统提示词 / 文档分页大小，见“文档挂载” |
| `Compaction` / `CompactionThreshold` / `CompactionPrompt` | 上下文压缩策略 / 触发比例（默认 0.8）/ 摘要提示词，见“上下文压缩” |
| `Reminder` | 附加在每条用户消息后的 `<reminder>` 内容 |
| `MaxRPCResultChars` | 单次 RPC 返回给模型的最大字符数（按字符计，中文不会被截成乱码）。超出部分会被截掉，并附上说明省略了多少字符；返回内容仍是合法 JSON。0 表示不限制 |
| `SkipSchema` | `NewClient` 不建表（自己用 `SchemaStatements` 做迁移） |
| `RecoverStaleAfter` | `NewClient` 只恢复超过这个时长没有心跳的会话，多实例部署时使用；0 表示恢复全部 |
| `StreamFlushInterval` | 流式输出时写库的间隔，默认 2 秒 |
| `StreamIdleTimeout` | 流式输出多久没有数据就中断，默认 2 分钟，负数关闭 |
| `KeepReasoning` | 把思考内容以 `reasoning_content` 回传给模型（DeepSeek 思考模式 + 工具调用需要）。无论开不开，思考内容都会存进 `Message.Reasoning` |
| `ExtraBody` | 合并进请求体，如 `temperature`、`top_p`、`reasoning_effort`；值为 `nil` 表示删除默认字段，例如不支持 `stream_options` 的服务可设 `"stream_options": nil` |
| `UseMaxCompletionTokens` | 用 `max_completion_tokens` 代替 `max_tokens`（新版 OpenAI 推理模型） |
| `Headers` / `HTTPClient` | 自定义请求头 / HTTP 客户端（默认不设整体超时，由 `StreamIdleTimeout` 兜底） |
| `MaxRetries` | 429 / 5xx / 网络错误重试次数，默认 2，负数关闭；已经开始输出后不再重试。代理在 HTTP 200 的流里返回的错误（如 `{"error":{"code":502}}`）也按其中的 code 判断是否重试 |
| `MaxSteps` | 单次运行最多 LLM 调用次数，默认 50 |
| `StaleAfter` | 运行锁心跳超时，默认 1 分钟 |
| `StopPollInterval` | 检查其他进程发来的 Stop 的间隔，默认 1 秒，负数关闭 |
| `RPCTimeout` | RPC 调用的默认超时时间，默认不超时 |
| `ToolConcurrency` | 同一轮 RPC 调用的并发数，0 = 全部并行（默认），1 = 逐个执行 |
| `BeforeLLMCall` | 每次调用大模型前的钩子，可用于预算控制 |
| `CacheControl` | 为 Claude 模型添加 `cache_control` 缓存断点 |
| `Logger` / `LogLevel` | 日志输出和级别，见“日志” |
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
