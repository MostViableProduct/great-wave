// Package compiler provides the top-level orchestrator that ties together
// cascade classification, Bayesian gating, keyword learning, belief states,
// and health scoring into a single coherent system.
//
// All external dependencies are expressed as narrow Go interfaces. Implement
// only the interfaces you need — the compiler gracefully degrades when
// optional adapters are nil (e.g., no LLM = pure heuristic mode).
package compiler

import (
	"context"
	"encoding/json"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/gate"
	"github.com/MostViableProduct/great-wave/pkg/health"
	"github.com/MostViableProduct/great-wave/pkg/keywords"
)

// LLMClassifier provides deep classification when heuristics are insufficient.
// The compiler calls this when the Bayesian gate determines heuristic confidence
// is too low. If nil, the compiler runs in pure-heuristic mode.
type LLMClassifier interface {
	Classify(ctx context.Context, content string, signalType string, categories []string) (*LLMResult, error)
}

// LLMResult holds the output of an LLM classification call.
type LLMResult struct {
	Category   string   `json:"category"`
	Confidence float64  `json:"confidence"`
	Keywords   []string `json:"keywords,omitempty"`
}

// VectorSearcher provides semantic similarity search for entity resolution.
// If nil, entity resolution is skipped and classification results are
// returned without related entities.
type VectorSearcher interface {
	Search(ctx context.Context, query string, limit int) ([]VectorMatch, error)
}

// VectorMatch represents a single similarity search result.
type VectorMatch struct {
	ID       string  `json:"id"`
	Score    float64 `json:"score"`
	Metadata map[string]string `json:"metadata,omitempty"`
}

// Embedder produces vector embeddings from text. Consumed by vector stores
// internally — not added to Deps since the vector store owns its embedder.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Dimensions() int
}

// EventSink receives domain events emitted by the compiler.
// If nil, events are silently discarded. Implementations may publish to
// message buses, webhooks, log files, or any other destination.
type EventSink interface {
	Emit(ctx context.Context, eventType string, payload json.RawMessage) error
}

// Signal represents an incoming signal to be classified and routed.
type Signal struct {
	// TenantID identifies the tenant (empty string for single-tenant deployments).
	TenantID string `json:"tenant_id,omitempty"`
	// Source identifies where the signal originated (e.g., "sentry", "prometheus").
	Source string `json:"source"`
	// Type is the signal type (e.g., "metric", "error", "deploy").
	Type string `json:"type"`
	// Content is the searchable text content of the signal.
	Content string `json:"content"`
	// Payload is the raw JSON payload for signal class inference.
	Payload json.RawMessage `json:"payload,omitempty"`
}

// Classification source constants for ClassifyResult.ClassificationSource.
const (
	SourceHeuristic      = "heuristic"
	SourceHeuristicGated = "heuristic_gated"
	SourceLLM            = "llm"
)

// ClassifyResult holds the full output of the cascade classification pipeline.
type ClassifyResult struct {
	Category             string                `json:"category"`
	RelevanceScore       float64               `json:"relevance_score"`
	ClassificationSource string                `json:"classification_source"`
	Confidence           float64               `json:"confidence"`
	SignalClass          classifier.SignalClass `json:"signal_class"`
	RelatedEntities      []string              `json:"related_entities,omitempty"`
}

// HealthResult holds the output of a health score computation.
type HealthResult struct {
	EntityID                string  `json:"entity_id"`
	Score                   float64 `json:"score"`
	ConfidenceIntervalLower float64 `json:"confidence_interval_lower"`
	ConfidenceIntervalUpper float64 `json:"confidence_interval_upper"`
}

// Deps bundles all external dependencies for the compiler.
// All fields are optional — set to nil to disable the corresponding feature.
type Deps struct {
	// LLM provides deep classification. Nil = pure heuristic mode.
	LLM LLMClassifier
	// Vector provides semantic search for entity resolution. Nil = no entity resolution.
	Vector VectorSearcher
	// Events receives domain events. Nil = events discarded.
	Events EventSink

	// Storage adapters for individual components.
	// Each can be nil for in-memory-only operation.
	GateStore    gate.GateStore
	HealthStore  health.HealthStore
	KeywordStore keywords.KeywordStore
}
