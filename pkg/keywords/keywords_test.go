package keywords

import (
	"errors"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
)

func testClassifier() *classifier.Classifier {
	return classifier.New(classifier.Config{
		Categories: []classifier.CategoryConfig{
			{
				Name:     "performance",
				Keywords: []string{"latency", "throughput", "p99", "slow", "timeout"},
			},
			{
				Name:     "reliability",
				Keywords: []string{"error", "failure", "crash", "outage"},
			},
		},
	})
}

// memKeywordStore implements KeywordStore for testing.
type memKeywordStore struct {
	keywords  map[string]string // keyword -> category
	promoted  []classifier.LearnedKeyword
	demotions int
	err       error
}

func newMemKeywordStore() *memKeywordStore {
	return &memKeywordStore{
		keywords: make(map[string]string),
	}
}

func (m *memKeywordStore) UpsertKeyword(keyword, category string) error {
	if m.err != nil {
		return m.err
	}
	m.keywords[keyword] = category
	return nil
}

func (m *memKeywordStore) PromoteKeywords(minConfidence float64, minObservations int) ([]classifier.LearnedKeyword, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.promoted, nil
}

func (m *memKeywordStore) DemoteKeywords(minConfidence float64, minObservations int) error {
	if m.err != nil {
		return m.err
	}
	m.demotions++
	return nil
}

func (m *memKeywordStore) LoadPromotedKeywords(limit int) ([]classifier.LearnedKeyword, error) {
	if m.err != nil {
		return nil, m.err
	}
	if limit < len(m.promoted) {
		return m.promoted[:limit], nil
	}
	return m.promoted, nil
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.MinConfidence != 0.7 {
		t.Errorf("Expected MinConfidence 0.7, got %v", cfg.MinConfidence)
	}
	if cfg.MinObservations != 10 {
		t.Errorf("Expected MinObservations 10, got %d", cfg.MinObservations)
	}
	if cfg.MaxPromoted != 200 {
		t.Errorf("Expected MaxPromoted 200, got %d", cfg.MaxPromoted)
	}
	if cfg.MaxExtracted != 5 {
		t.Errorf("Expected MaxExtracted 5, got %d", cfg.MaxExtracted)
	}
}

func TestNewLearner_DefaultsZeroValues(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(Config{}, nil, cls, runtime)

	if l.config.MinConfidence != 0.7 {
		t.Errorf("Expected default MinConfidence 0.7, got %v", l.config.MinConfidence)
	}
	if l.config.MinObservations != 10 {
		t.Errorf("Expected default MinObservations 10, got %d", l.config.MinObservations)
	}
	if l.config.MaxPromoted != 200 {
		t.Errorf("Expected default MaxPromoted 200, got %d", l.config.MaxPromoted)
	}
	if l.config.MaxExtracted != 5 {
		t.Errorf("Expected default MaxExtracted 5, got %d", l.config.MaxExtracted)
	}
}

func TestTokenize(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    []string
	}{
		{"empty", "", nil},
		{"single_word", "Hello", []string{"hello"}},
		{"multiple_words", "Hello World", []string{"hello", "world"}},
		{"punctuation", "error: connection_refused!", []string{"error", "connection_refused"}},
		{"mixed_case", "CPU Memory DISK", []string{"cpu", "memory", "disk"}},
		{"numbers", "p99 latency 500ms", []string{"p99", "latency", "500ms"}},
		{"underscores", "response_time error_rate", []string{"response_time", "error_rate"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Tokenize(tt.content)
			if len(got) != len(tt.want) {
				t.Errorf("Tokenize(%q) = %v, want %v", tt.content, got, tt.want)
				return
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Errorf("Tokenize(%q)[%d] = %q, want %q", tt.content, i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestExtractKeywords(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	// "kubernetes" and "pod" are novel; "latency" and "error" are in static keywords
	keywords := l.ExtractKeywords("kubernetes pod latency error detected", "performance")

	found := make(map[string]bool)
	for _, kw := range keywords {
		found[kw] = true
	}

	if !found["kubernetes"] {
		t.Error("Expected 'kubernetes' to be extracted as novel keyword")
	}
	if !found["pod"] {
		t.Error("Expected 'pod' to be extracted as novel keyword")
	}
	if !found["detected"] {
		t.Error("Expected 'detected' to be extracted as novel keyword")
	}
	if found["latency"] {
		t.Error("'latency' should be filtered as a static keyword")
	}
	if found["error"] {
		t.Error("'error' should be filtered as a static keyword")
	}
}

func TestExtractKeywords_FiltersStopwords(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	keywords := l.ExtractKeywords("the service has been down for hours", "reliability")

	found := make(map[string]bool)
	for _, kw := range keywords {
		found[kw] = true
	}

	if found["the"] {
		t.Error("'the' should be filtered as a stopword")
	}
	if found["has"] {
		t.Error("'has' should be filtered as a stopword")
	}
	if found["been"] {
		t.Error("'been' should be filtered as a stopword")
	}
	if found["for"] {
		t.Error("'for' should be filtered as a stopword")
	}
	if !found["service"] {
		t.Error("Expected 'service' to be extracted")
	}
}

func TestExtractKeywords_FiltersShortTokens(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	keywords := l.ExtractKeywords("db is up ok", "reliability")

	for _, kw := range keywords {
		if len(kw) < 3 {
			t.Errorf("Short token %q should have been filtered", kw)
		}
	}
}

func TestExtractKeywords_MaxExtracted(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	cfg := DefaultConfig()
	cfg.MaxExtracted = 2
	l := NewLearner(cfg, nil, cls, runtime)

	keywords := l.ExtractKeywords("kubernetes pod restart container deployment rollout", "deployment")

	if len(keywords) > 2 {
		t.Errorf("Expected at most 2 keywords, got %d: %v", len(keywords), keywords)
	}
}

func TestExtractKeywords_NoDuplicates(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	keywords := l.ExtractKeywords("kubernetes kubernetes kubernetes pod pod", "deployment")

	seen := make(map[string]bool)
	for _, kw := range keywords {
		if seen[kw] {
			t.Errorf("Duplicate keyword found: %q", kw)
		}
		seen[kw] = true
	}
}

func TestRecordDisagreement(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	store := newMemKeywordStore()
	l := NewLearner(DefaultConfig(), store, cls, runtime)

	l.RecordDisagreement("kubernetes pod restart detected", "deployment", 0.9)

	if len(store.keywords) == 0 {
		t.Error("Expected keywords to be upserted")
	}
	if _, ok := store.keywords["kubernetes"]; !ok {
		t.Error("Expected 'kubernetes' to be upserted")
	}
}

func TestRecordDisagreement_LowConfidence(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	store := newMemKeywordStore()
	l := NewLearner(DefaultConfig(), store, cls, runtime)

	l.RecordDisagreement("kubernetes pod restart", "deployment", 0.3)

	if len(store.keywords) != 0 {
		t.Error("Should not upsert keywords when LLM confidence is below threshold")
	}
}

func TestRecordDisagreement_NilStore(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	// Should not panic with nil store
	l.RecordDisagreement("kubernetes pod restart", "deployment", 0.9)
}

func TestPromote(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	store := newMemKeywordStore()
	store.promoted = []classifier.LearnedKeyword{
		{Keyword: "k8s", Category: "deployment", Weight: 1.5, Confidence: 0.9},
		{Keyword: "oom", Category: "reliability", Weight: 2.0, Confidence: 0.85},
	}

	l := NewLearner(DefaultConfig(), store, cls, runtime)

	count, err := l.Promote()
	if err != nil {
		t.Fatalf("Promote failed: %v", err)
	}
	if count != 2 {
		t.Errorf("Expected 2 promoted, got %d", count)
	}
	if runtime.Count() != 2 {
		t.Errorf("Expected runtime to have 2 keywords, got %d", runtime.Count())
	}
	if store.demotions != 1 {
		t.Errorf("Expected 1 demotion call, got %d", store.demotions)
	}
}

func TestPromote_NilStore(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	count, err := l.Promote()
	if err != nil {
		t.Fatalf("Promote with nil store should not error: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected 0 with nil store, got %d", count)
	}
}

func TestPromote_StoreError(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	store := newMemKeywordStore()
	store.err = errors.New("database unavailable")

	l := NewLearner(DefaultConfig(), store, cls, runtime)

	_, err := l.Promote()
	if err == nil {
		t.Error("Expected error from store")
	}
}

func TestLoadPromoted(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	store := newMemKeywordStore()
	store.promoted = []classifier.LearnedKeyword{
		{Keyword: "k8s", Category: "deployment", Weight: 1.5},
		{Keyword: "pod", Category: "deployment", Weight: 1.0},
		{Keyword: "oom", Category: "reliability", Weight: 2.0},
	}

	l := NewLearner(DefaultConfig(), store, cls, runtime)

	count, err := l.LoadPromoted()
	if err != nil {
		t.Fatalf("LoadPromoted failed: %v", err)
	}
	if count != 3 {
		t.Errorf("Expected 3 loaded, got %d", count)
	}
	if runtime.Count() != 3 {
		t.Errorf("Expected runtime to have 3 keywords, got %d", runtime.Count())
	}
}

func TestLoadPromoted_NilStore(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	l := NewLearner(DefaultConfig(), nil, cls, runtime)

	count, err := l.LoadPromoted()
	if err != nil {
		t.Fatalf("LoadPromoted with nil store should not error: %v", err)
	}
	if count != 0 {
		t.Errorf("Expected 0 with nil store, got %d", count)
	}
}

func TestStopwords(t *testing.T) {
	expected := []string{"the", "and", "for", "with", "this", "that", "from", "not", "but", "all", "can", "use"}
	for _, word := range expected {
		if !Stopwords[word] {
			t.Errorf("Expected %q to be a stopword", word)
		}
	}

	nonStopwords := []string{"kubernetes", "deploy", "latency", "server", "database"}
	for _, word := range nonStopwords {
		if Stopwords[word] {
			t.Errorf("%q should not be a stopword", word)
		}
	}
}

func TestEndToEnd_LearnAndPromote(t *testing.T) {
	cls := testClassifier()
	runtime := classifier.NewLearnedKeywordStore()
	store := newMemKeywordStore()
	l := NewLearner(DefaultConfig(), store, cls, runtime)

	// Simulate disagreements — LLM says "deployment" for content with novel keywords
	l.RecordDisagreement("kubernetes pod restart container", "deployment", 0.9)
	l.RecordDisagreement("rollout canary deployment strategy", "deployment", 0.85)

	// Verify keywords were recorded
	if len(store.keywords) == 0 {
		t.Fatal("Expected keywords to be recorded")
	}

	// Simulate promotion (store returns high-confidence keywords)
	store.promoted = []classifier.LearnedKeyword{
		{Keyword: "kubernetes", Category: "deployment", Weight: 1.5, Confidence: 0.9},
		{Keyword: "pod", Category: "deployment", Weight: 1.0, Confidence: 0.8},
	}

	count, err := l.Promote()
	if err != nil {
		t.Fatalf("Promote failed: %v", err)
	}
	if count != 2 {
		t.Errorf("Expected 2 promoted, got %d", count)
	}

	// Verify runtime store was updated — classifier can now use learned keywords
	result := cls.ClassifyWithLearned("kubernetes pod restart detected", "event", runtime)
	if result.Category != "deployment" {
		t.Errorf("Expected deployment after promotion, got %s", result.Category)
	}
}
