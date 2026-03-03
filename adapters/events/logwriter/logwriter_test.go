package logwriter

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestEmit_WritesJSON(t *testing.T) {
	var buf bytes.Buffer
	sink := New(&buf)

	payload := json.RawMessage(`{"key":"value"}`)
	err := sink.Emit(context.Background(), "test.event", payload)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw: %s", err, buf.String())
	}

	if entry["event_type"] != "test.event" {
		t.Errorf("expected event_type test.event, got %v", entry["event_type"])
	}

	p := entry["payload"].(map[string]any)
	if p["key"] != "value" {
		t.Errorf("expected payload.key=value, got %v", p["key"])
	}

	ts, ok := entry["timestamp"].(string)
	if !ok || ts == "" {
		t.Error("expected timestamp to be set")
	}

	// Verify RFC3339 format
	if _, err := time.Parse(time.RFC3339, ts); err != nil {
		t.Errorf("timestamp not RFC3339: %s", ts)
	}
}

func TestEmit_NewlineDelimited(t *testing.T) {
	var buf bytes.Buffer
	sink := New(&buf)

	sink.Emit(context.Background(), "event.one", json.RawMessage(`{}`))
	sink.Emit(context.Background(), "event.two", json.RawMessage(`{}`))

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 2 {
		t.Errorf("expected 2 lines, got %d", len(lines))
	}
}

func TestEmit_Concurrent(t *testing.T) {
	var buf bytes.Buffer
	sink := New(&buf)

	var wg sync.WaitGroup
	n := 50
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			sink.Emit(context.Background(), "concurrent.event", json.RawMessage(`{"i":1}`))
		}()
	}
	wg.Wait()

	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != n {
		t.Errorf("expected %d lines, got %d", n, len(lines))
	}

	// Each line should be valid JSON
	for i, line := range lines {
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			t.Errorf("line %d is not valid JSON: %v", i, err)
		}
	}
}
