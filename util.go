package agent

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"time"
	"unicode/utf8"
)

func newID(prefix string) string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return prefix + "_" + hex.EncodeToString(b[:])
}

func nowMillis() int64 { return time.Now().UnixMilli() }

func fromMillis(ms int64) time.Time { return time.UnixMilli(ms) }

func rawOrNil(s string) json.RawMessage {
	if s == "" {
		return nil
	}
	return json.RawMessage(s)
}

func boolInt(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// marshalJSON encodes without HTML escaping so stored raw JSON stays readable
// and identical on every replay.
func marshalJSON(v any) ([]byte, error) {
	var buf bytes.Buffer
	enc := json.NewEncoder(&buf)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(v); err != nil {
		return nil, err
	}
	return bytes.TrimRight(buf.Bytes(), "\n"), nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// truncateRunes cuts s to at most n characters (not bytes, so multi-byte
// text is never split) and reports how many characters were dropped.
func truncateRunes(s string, n int) (string, int) {
	if utf8.RuneCountInString(s) <= n {
		return s, 0
	}
	r := []rune(s)
	return string(r[:n]), len(r) - n
}
