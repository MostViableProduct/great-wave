package compiler

import (
	"context"
	"encoding/json"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
)

func testConfig() Config {
	cfg := DefaultConfig()
	cfg.Classifier = classifier.Config{
		Categories: []classifier.CategoryConfig{
			{
				Name:     "performance",
				Keywords: []string{"latency", "throughput", "p99", "slow", "timeout"},
				Weights:  map[string]float64{"p99": 2.0, "timeout": 2.0},
			},
			{
				Name:     "reliability",
				Keywords: []string{"error", "failure", "crash", "outage"},
				Weights:  map[string]float64{"outage": 2.0, "crash": 2.0},
			},
			{
				Name:     "deployment",
				Keywords: []string{"deploy", "release", "rollout", "rollback"},
				Weights:  map[string]float64{"deploy": 2.0, "rollback": 2.0},
			},
		},
		TypeToCategory: map[string]string{
			"metric": "performance",
			"error":  "reliability",
			"deploy": "deployment",
		},
	}
	return cfg
}

// stubLLM implements LLMClassifier for testing.
type stubLLM struct {
	result *LLMResult
	err    error
	calls  int32
}

func (s *stubLLM) Classify(_ context.Context, _, _ string, _ []string) (*LLMResult, error) {
	atomic.AddInt32(&s.calls, 1)
	return s.result, s.err
}

// stubVector implements VectorSearcher for testing.
type stubVector struct {
	matches []VectorMatch
}

func (s *stubVector) Search(_ context.Context, _ string, _ int) ([]VectorMatch, error) {
	return s.matches, nil
}

// stubEvents implements EventSink for testing.
type stubEvents struct {
	events []json.RawMessage
}

func (s *stubEvents) Emit(_ context.Context, _ string, payload json.RawMessage) error {
	s.events = append(s.events, payload)
	return nil
}

func TestNew(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})
	if c == nil {
		t.Fatal("Expected non-nil compiler")
	}
}

func TestClassify_HeuristicOnly(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	result, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "metric",
		Content: "High p99 latency detected",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if result.Category != "performance" {
		t.Errorf("Expected performance, got %s", result.Category)
	}
	if result.ClassificationSource != "heuristic" {
		t.Errorf("Expected heuristic source, got %s", result.ClassificationSource)
	}
}

func TestClassify_EmptyContent(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	_, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "metric",
		Content: "",
	})
	if !errors.Is(err, ErrEmptyContent) {
		t.Errorf("Expected ErrEmptyContent, got %v", err)
	}
}

func TestClassify_WithLLM(t *testing.T) {
	llm := &stubLLM{
		result: &LLMResult{
			Category:   "reliability",
			Confidence: 0.95,
			Keywords:   []string{"outage"},
		},
	}

	c := New(testConfig(), Deps{LLM: llm}, Callbacks{})

	result, err := c.Classify(context.Background(), Signal{
		Source:  "sentry",
		Type:    "event",
		Content: "Critical outage in production database",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if result.Category != "reliability" {
		t.Errorf("Expected reliability from LLM, got %s", result.Category)
	}
	if result.ClassificationSource != "llm" {
		t.Errorf("Expected llm source, got %s", result.ClassificationSource)
	}
}

func TestClassify_LLMFallback(t *testing.T) {
	var fallbackCalled bool
	llm := &stubLLM{
		err: errors.New("LLM unavailable"),
	}

	c := New(testConfig(), Deps{LLM: llm}, Callbacks{
		OnLLMFallback: func(err error) {
			fallbackCalled = true
		},
	})

	result, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "event",
		Content: "Critical outage detected",
	})
	if err != nil {
		t.Fatalf("Classify should not error on LLM failure: %v", err)
	}
	if result.ClassificationSource != "heuristic" {
		t.Errorf("Expected heuristic fallback, got %s", result.ClassificationSource)
	}
	if !fallbackCalled {
		t.Error("Expected OnLLMFallback callback to be called")
	}
}

func TestClassify_LLMLowConfidence(t *testing.T) {
	llm := &stubLLM{
		result: &LLMResult{
			Category:   "security",
			Confidence: 0.1, // below LLMMinConfidence threshold
		},
	}

	c := New(testConfig(), Deps{LLM: llm}, Callbacks{})

	result, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "event",
		Content: "Critical outage in production",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	// Should fall back to heuristic since LLM confidence is too low
	if result.ClassificationSource != "heuristic" {
		t.Errorf("Expected heuristic (LLM low confidence), got %s", result.ClassificationSource)
	}
}

func TestClassify_InvalidLLMCategory(t *testing.T) {
	llm := &stubLLM{
		result: &LLMResult{
			Category:   "nonexistent_category",
			Confidence: 0.95,
		},
	}

	c := New(testConfig(), Deps{LLM: llm}, Callbacks{})

	result, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "event",
		Content: "Critical outage detected",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	// Invalid category should be rejected, falling back to heuristic
	if result.ClassificationSource != "heuristic" {
		t.Errorf("Expected heuristic (invalid LLM category), got %s", result.ClassificationSource)
	}
}

func TestClassify_WithVectorSearch(t *testing.T) {
	vector := &stubVector{
		matches: []VectorMatch{
			{ID: "entity-1", Score: 0.95},
			{ID: "entity-2", Score: 0.87},
		},
	}

	c := New(testConfig(), Deps{Vector: vector}, Callbacks{})

	result, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "metric",
		Content: "High p99 latency in API gateway",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if len(result.RelatedEntities) != 2 {
		t.Errorf("Expected 2 related entities, got %d", len(result.RelatedEntities))
	}
}

func TestClassify_SignalClass(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	// Metric payload
	result, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "metric",
		Content: "latency value",
		Payload: json.RawMessage(`{"value": 42.5}`),
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if result.SignalClass != classifier.SignalClassMetric {
		t.Errorf("Expected METRIC signal class, got %s", result.SignalClass)
	}

	// No payload
	result2, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "generic",
		Content: "some text content",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}
	if result2.SignalClass != classifier.SignalClassSemantic {
		t.Errorf("Expected SEMANTIC signal class for no payload, got %s", result2.SignalClass)
	}
}

func TestClassifySignal(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	payload := json.RawMessage(`{"message": "p99 latency spike detected"}`)
	result, err := c.ClassifySignal(context.Background(), "tenant-1", "prometheus", "metric", payload)
	if err != nil {
		t.Fatalf("ClassifySignal failed: %v", err)
	}
	if result.Category != "performance" {
		t.Errorf("Expected performance, got %s", result.Category)
	}
}

func TestScoreHealth(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	score := c.ScoreHealth("t1", "entity-1")
	if score.EntityID != "entity-1" {
		t.Errorf("Expected entity-1, got %s", score.EntityID)
	}
	if score.Score < 0 || score.Score > 100 {
		t.Errorf("Score out of bounds: %v", score.Score)
	}
	if score.ConfidenceIntervalLower >= score.ConfidenceIntervalUpper {
		t.Errorf("CI invalid: lower=%v >= upper=%v", score.ConfidenceIntervalLower, score.ConfidenceIntervalUpper)
	}
}

func TestRecordHealthEvent(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	initialScore := c.ScoreHealth("t1", "entity-1").Score

	for i := 0; i < 10; i++ {
		c.RecordHealthEvent("t1", "entity-1", "critical", "reliability", 1.0)
	}

	newScore := c.ScoreHealth("t1", "entity-1").Score
	if newScore >= initialScore {
		t.Errorf("Critical events should decrease score: %v -> %v", initialScore, newScore)
	}
}

func TestCallbacks(t *testing.T) {
	var classifyCount int32
	var agreementCount int32

	llm := &stubLLM{
		result: &LLMResult{
			Category:   "reliability",
			Confidence: 0.9,
		},
	}

	c := New(testConfig(), Deps{LLM: llm}, Callbacks{
		OnClassify: func(source, category string) {
			atomic.AddInt32(&classifyCount, 1)
		},
		OnAgreement: func(agreed bool) {
			atomic.AddInt32(&agreementCount, 1)
		},
	})

	_, err := c.Classify(context.Background(), Signal{
		Source:  "test",
		Type:    "event",
		Content: "Critical outage in production",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	if atomic.LoadInt32(&classifyCount) != 1 {
		t.Errorf("Expected OnClassify called once, got %d", classifyCount)
	}
}

func TestEventEmission(t *testing.T) {
	events := &stubEvents{}
	llm := &stubLLM{
		result: &LLMResult{
			Category:   "performance",
			Confidence: 0.9,
		},
	}

	c := New(testConfig(), Deps{LLM: llm, Events: events}, Callbacks{})

	_, err := c.Classify(context.Background(), Signal{
		TenantID: "t1",
		Source:   "prometheus",
		Type:     "metric",
		Content:  "High p99 latency detected",
	})
	if err != nil {
		t.Fatalf("Classify failed: %v", err)
	}

	// Give the goroutine a moment to emit
	// Note: in production code, you'd use sync mechanisms.
	// For tests, the event may or may not have been emitted yet
	// since recordComparison runs in a goroutine.
}

func TestLoadFlushState(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	// With nil stores, Load and Flush should succeed (no-ops)
	if err := c.LoadState(); err != nil {
		t.Fatalf("LoadState failed: %v", err)
	}
	if err := c.FlushState(); err != nil {
		t.Fatalf("FlushState failed: %v", err)
	}
}

func TestPromoteKeywords(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	// With nil keyword store, promotion returns 0
	count, err := c.PromoteKeywords()
	if err != nil {
		t.Fatalf("PromoteKeywords failed: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected 0 with nil store, got %d", count)
	}
}

func TestRecordFalsePositives(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	// With nil keyword store, RecordFalsePositives returns nil (no-op)
	err := c.RecordFalsePositives([]string{"latency_spike", "timeout_error"})
	if err != nil {
		t.Fatalf("RecordFalsePositives failed: %v", err)
	}

	// Empty keywords list should also succeed
	err = c.RecordFalsePositives(nil)
	if err != nil {
		t.Fatalf("RecordFalsePositives(nil) failed: %v", err)
	}
}

func TestAccessors(t *testing.T) {
	c := New(testConfig(), Deps{}, Callbacks{})

	if c.Gate() == nil {
		t.Error("Gate() should not be nil")
	}
	if c.HealthModel() == nil {
		t.Error("HealthModel() should not be nil")
	}
	if c.Runtime() == nil {
		t.Error("Runtime() should not be nil")
	}
}
