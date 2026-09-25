package examples

import (
	"context"
	"database/sql"
	"path/filepath"
	"testing"

	agent "github.com/zxcHolmes/agent-go"
)

// A database created by v0.6.1 (no ref_seq column) keeps working after upgrade.
func TestMigrateFromV061(t *testing.T) {
	db, err := sql.Open("sqlite", "file:"+filepath.Join(t.TempDir(), "old.db"))
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	ctx := context.Background()
	if _, err := db.ExecContext(ctx, `CREATE TABLE agent_messages (
		id VARCHAR(64) NOT NULL PRIMARY KEY, session_id VARCHAR(64) NOT NULL, seq BIGINT NOT NULL,
		role VARCHAR(32) NOT NULL, kind VARCHAR(32) NOT NULL, status VARCHAR(32) NOT NULL,
		content TEXT NOT NULL, reasoning TEXT NOT NULL, tool_call_id VARCHAR(255) NOT NULL,
		raw TEXT NOT NULL, created_at BIGINT NOT NULL, updated_at BIGINT NOT NULL, UNIQUE (session_id, seq))`); err != nil {
		t.Fatal(err)
	}
	if _, err := db.ExecContext(ctx, `INSERT INTO agent_messages VALUES ('m_old','s',1,'user','','done','hi','','','{"role":"user","content":"hi"}',1,1)`); err != nil {
		t.Fatal(err)
	}
	store := agent.NewSQLStore(db, agent.SQLite)
	for i := 0; i < 2; i++ { // idempotent
		if _, err := agent.NewClient(ctx, agent.Config{Store: store}); err != nil {
			t.Fatal(err)
		}
	}
	var refSeq int64
	if err := db.QueryRowContext(ctx, "SELECT ref_seq FROM agent_messages WHERE id = 'm_old'").Scan(&refSeq); err != nil || refSeq != 0 {
		t.Fatalf("ref_seq=%d err=%v", refSeq, err)
	}
}
