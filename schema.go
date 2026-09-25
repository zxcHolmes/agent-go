package agent

import (
	"fmt"
	"strings"
)

type tableDef struct {
	name    string
	columns []string
	indexes [][2]string // name, column list
}

func tableDefs(text string) []tableDef {
	return []tableDef{
		{
			name:    "agent_sessions",
			indexes: [][2]string{{"idx_agent_sessions_status", "status, updated_at"}},
			columns: []string{
				"id VARCHAR(64) NOT NULL PRIMARY KEY",
				"status VARCHAR(32) NOT NULL",
				"run_id VARCHAR(64) NOT NULL",
				"last_error " + text + " NOT NULL",
				"metadata " + text + " NOT NULL",
				"created_at BIGINT NOT NULL",
				"updated_at BIGINT NOT NULL",
			},
		},
		{
			name: "agent_messages",
			columns: []string{
				"id VARCHAR(64) NOT NULL PRIMARY KEY",
				"session_id VARCHAR(64) NOT NULL",
				"seq BIGINT NOT NULL",
				"role VARCHAR(32) NOT NULL",
				"kind VARCHAR(32) NOT NULL",
				"ref_seq BIGINT NOT NULL",
				"status VARCHAR(32) NOT NULL",
				"content " + text + " NOT NULL",
				"reasoning " + text + " NOT NULL",
				"tool_call_id VARCHAR(255) NOT NULL",
				"raw " + text + " NOT NULL",
				"created_at BIGINT NOT NULL",
				"updated_at BIGINT NOT NULL",
				"UNIQUE (session_id, seq)",
			},
		},
		{
			name: "agent_llm_calls",
			columns: []string{
				"id VARCHAR(64) NOT NULL PRIMARY KEY",
				"session_id VARCHAR(64) NOT NULL",
				"message_id VARCHAR(64) NOT NULL",
				"model VARCHAR(255) NOT NULL",
				"response_id VARCHAR(255) NOT NULL",
				"finish_reason VARCHAR(64) NOT NULL",
				"prompt_tokens BIGINT NOT NULL",
				"completion_tokens BIGINT NOT NULL",
				"total_tokens BIGINT NOT NULL",
				"cached_tokens BIGINT NOT NULL",
				"cache_write_tokens BIGINT NOT NULL",
				"reasoning_tokens BIGINT NOT NULL",
				"audio_input_tokens BIGINT NOT NULL",
				"audio_output_tokens BIGINT NOT NULL",
				"accepted_prediction_tokens BIGINT NOT NULL",
				"rejected_prediction_tokens BIGINT NOT NULL",
				"raw_usage " + text + " NOT NULL",
				"input_credits DOUBLE PRECISION NOT NULL",
				"cached_credits DOUBLE PRECISION NOT NULL",
				"cache_write_credits DOUBLE PRECISION NOT NULL",
				"output_credits DOUBLE PRECISION NOT NULL",
				"total_credits DOUBLE PRECISION NOT NULL",
				"latency_ms BIGINT NOT NULL",
				"bill_id VARCHAR(64) NOT NULL",
				"claimed_at BIGINT NOT NULL",
				"billed_at BIGINT NOT NULL",
				"created_at BIGINT NOT NULL",
			},
			indexes: [][2]string{
				{"idx_agent_llm_calls_session", "session_id, created_at"},
				{"idx_agent_llm_calls_unbilled", "billed_at, session_id"},
				{"idx_agent_llm_calls_bill", "bill_id"},
			},
		},
		{
			name: "agent_queued_messages",
			columns: []string{
				"id VARCHAR(64) NOT NULL PRIMARY KEY",
				"session_id VARCHAR(64) NOT NULL",
				"content " + text + " NOT NULL",
				"raw " + text + " NOT NULL",
				"created_at BIGINT NOT NULL",
			},
			indexes: [][2]string{{"idx_agent_queued_messages_session", "session_id, created_at"}},
		},
		{
			name: "agent_rpc_calls",
			columns: []string{
				"id VARCHAR(64) NOT NULL PRIMARY KEY",
				"session_id VARCHAR(64) NOT NULL",
				"message_id VARCHAR(64) NOT NULL",
				"tool_call_id VARCHAR(255) NOT NULL",
				"call_index BIGINT NOT NULL",
				"method VARCHAR(255) NOT NULL",
				"params " + text + " NOT NULL",
				"rpc_id " + text + " NOT NULL",
				"require_confirm BIGINT NOT NULL",
				"status VARCHAR(32) NOT NULL",
				"result " + text + " NOT NULL",
				"result_message_id VARCHAR(64) NOT NULL",
				"created_at BIGINT NOT NULL",
				"updated_at BIGINT NOT NULL",
			},
			indexes: [][2]string{
				{"idx_agent_rpc_calls_status", "session_id, status"},
				{"idx_agent_rpc_calls_message", "message_id"},
			},
		},
	}
}

// SchemaStatements returns the DDL Init executes for a dialect, for callers
// who prefer to run migrations with their own tooling.
func SchemaStatements(d Dialect) []string {
	text := "TEXT"
	if d == MySQL {
		text = "LONGTEXT"
	}
	var stmts []string
	for _, t := range tableDefs(text) {
		cols := append([]string{}, t.columns...)
		if d == MySQL {
			// MySQL has no CREATE INDEX IF NOT EXISTS, so declare indexes inline.
			for _, ix := range t.indexes {
				cols = append(cols, fmt.Sprintf("KEY %s (%s)", ix[0], ix[1]))
			}
		}
		stmt := fmt.Sprintf("CREATE TABLE IF NOT EXISTS %s (\n  %s\n)", t.name, strings.Join(cols, ",\n  "))
		if d == MySQL {
			stmt += " ENGINE=InnoDB DEFAULT CHARSET=utf8mb4"
		}
		stmts = append(stmts, stmt)
		if d != MySQL {
			for _, ix := range t.indexes {
				stmts = append(stmts, fmt.Sprintf("CREATE INDEX IF NOT EXISTS %s ON %s (%s)", ix[0], t.name, ix[1]))
			}
		}
	}
	return stmts
}
