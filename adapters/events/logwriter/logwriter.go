// Package logwriter implements compiler.EventSink by writing JSON events to an io.Writer.
//
// Useful for development, debugging, and as a reference for building
// NATS/Kafka/webhook sinks.
//
// Usage:
//
//	sink := logwriter.New(os.Stdout)
//	sink.Emit(ctx, "classification.compared", payload)
package logwriter

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"time"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

// Compile-time interface check.
var _ compiler.EventSink = (*Sink)(nil)

// Sink writes domain events as JSON lines to the underlying writer.
type Sink struct {
	w  io.Writer
	mu sync.Mutex
}

// New creates a Sink that writes to w.
func New(w io.Writer) *Sink {
	return &Sink{w: w}
}

type logEntry struct {
	EventType string          `json:"event_type"`
	Payload   json.RawMessage `json:"payload"`
	Timestamp string          `json:"timestamp"`
}

// Emit writes a JSON line for the event. Thread-safe.
func (s *Sink) Emit(_ context.Context, eventType string, payload json.RawMessage) error {
	entry := logEntry{
		EventType: eventType,
		Payload:   payload,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	data = append(data, '\n')

	s.mu.Lock()
	defer s.mu.Unlock()
	_, err = s.w.Write(data)
	return err
}
