package agent

import (
	"context"
	"database/sql"
	"strconv"
	"strings"
)

// Dialect selects the SQL flavour used for DDL and placeholders.
// All queries issued by the SDK are written with '?' placeholders; stores for
// dialects that use another style (Postgres) rewrite them.
type Dialect string

const (
	SQLite   Dialect = "sqlite"
	Postgres Dialect = "postgres"
	MySQL    Dialect = "mysql"
)

// Rows is the subset of *sql.Rows the SDK needs.
type Rows interface {
	Next() bool
	Scan(dest ...any) error
	Err() error
	Close() error
}

// Store is the storage abstraction used by the SDK. It mirrors plain SQL
// execution so any database can be plugged in by implementing three methods.
// Use NewSQLStore to adapt a database/sql handle.
type Store interface {
	// Dialect reports which SQL flavour the store speaks.
	Dialect() Dialect
	// Exec runs a statement and returns the number of affected rows
	// (0 when the driver cannot report it).
	Exec(ctx context.Context, query string, args ...any) (int64, error)
	// Query runs a query and returns its rows.
	Query(ctx context.Context, query string, args ...any) (Rows, error)
}

// SQLExecutor is satisfied by *sql.DB, *sql.Tx and *sql.Conn.
type SQLExecutor interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
}

// SQLStore is the Store provider backed by database/sql. Bring your own driver
// (e.g. modernc.org/sqlite, github.com/jackc/pgx/v5/stdlib,
// github.com/go-sql-driver/mysql).
type SQLStore struct {
	db      SQLExecutor
	dialect Dialect
}

// NewSQLStore wraps a database/sql handle.
func NewSQLStore(db SQLExecutor, dialect Dialect) *SQLStore {
	return &SQLStore{db: db, dialect: dialect}
}

func (s *SQLStore) Dialect() Dialect { return s.dialect }

func (s *SQLStore) Exec(ctx context.Context, query string, args ...any) (int64, error) {
	res, err := s.db.ExecContext(ctx, s.rebind(query), args...)
	if err != nil {
		return 0, err
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, nil
	}
	return n, nil
}

func (s *SQLStore) Query(ctx context.Context, query string, args ...any) (Rows, error) {
	rows, err := s.db.QueryContext(ctx, s.rebind(query), args...)
	if err != nil {
		return nil, err
	}
	return rows, nil
}

// rebind converts '?' placeholders to '$n' for Postgres. SDK queries never
// contain '?' inside string literals, so a plain scan is sufficient.
func (s *SQLStore) rebind(q string) string {
	if s.dialect != Postgres {
		return q
	}
	var b strings.Builder
	b.Grow(len(q) + 8)
	n := 0
	for i := 0; i < len(q); i++ {
		if q[i] == '?' {
			n++
			b.WriteByte('$')
			b.WriteString(strconv.Itoa(n))
			continue
		}
		b.WriteByte(q[i])
	}
	return b.String()
}
