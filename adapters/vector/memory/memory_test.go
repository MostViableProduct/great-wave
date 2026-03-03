package memoryvec

import (
	"context"
	"math"
	"sync"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

// stubEmbedder returns deterministic unit vectors for testing.
type stubEmbedder struct {
	vectors map[string][]float32
	dims    int
}

var _ compiler.Embedder = (*stubEmbedder)(nil)

func (s *stubEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i, t := range texts {
		if v, ok := s.vectors[t]; ok {
			result[i] = v
		} else {
			// Default: zero vector
			result[i] = make([]float32, s.dims)
		}
	}
	return result, nil
}

func (s *stubEmbedder) Dimensions() int { return s.dims }

func newTestEmbedder() *stubEmbedder {
	return &stubEmbedder{
		dims: 3,
		vectors: map[string][]float32{
			"performance signal": {1, 0, 0},
			"reliability signal": {0, 1, 0},
			"deployment signal":  {0, 0, 1},
			"perf query":         {0.9, 0.1, 0},   // close to performance
			"mixed signal":       {0.5, 0.5, 0},    // between performance and reliability
			"replace-me":         {0.3, 0.3, 0.3},  // initial vector
			"replace-me-v2":      {1.0, 0.0, 0.0},  // replacement vector
		},
	}
}

func TestAddAndSearch(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	err := store.AddTexts(context.Background(),
		[]string{"perf-1", "rel-1", "dep-1"},
		[]string{"performance signal", "reliability signal", "deployment signal"},
		nil,
	)
	if err != nil {
		t.Fatalf("AddTexts failed: %v", err)
	}

	if store.Len() != 3 {
		t.Fatalf("expected 3 entries, got %d", store.Len())
	}

	matches, err := store.Search(context.Background(), "perf query", 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}
	if matches[0].ID != "perf-1" {
		t.Errorf("expected perf-1 as top match, got %s", matches[0].ID)
	}
	if matches[0].Score <= matches[1].Score {
		t.Errorf("expected scores in descending order: %f <= %f", matches[0].Score, matches[1].Score)
	}
}

func TestReplaceByID(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	// Add initial entry
	store.Add(Entry{ID: "entry-1", Vector: emb.vectors["replace-me"]})
	if store.Len() != 1 {
		t.Fatalf("expected 1 entry, got %d", store.Len())
	}

	// Replace with new vector
	store.Add(Entry{ID: "entry-1", Vector: emb.vectors["replace-me-v2"]})
	if store.Len() != 1 {
		t.Fatalf("expected 1 entry after replace, got %d", store.Len())
	}

	// Search should find the replaced vector
	matches, err := store.Search(context.Background(), "performance signal", 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	// replaced vector is [1,0,0] which matches "performance signal" perfectly
	if matches[0].Score < 0.99 {
		t.Errorf("expected score ~1.0 after replacement, got %f", matches[0].Score)
	}
}

func TestDelete(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	store.Add(
		Entry{ID: "a", Vector: []float32{1, 0, 0}},
		Entry{ID: "b", Vector: []float32{0, 1, 0}},
		Entry{ID: "c", Vector: []float32{0, 0, 1}},
	)

	if store.Len() != 3 {
		t.Fatalf("expected 3, got %d", store.Len())
	}

	store.Delete("b")
	if store.Len() != 2 {
		t.Fatalf("expected 2 after delete, got %d", store.Len())
	}

	// Delete non-existent ID should be a no-op
	store.Delete("nonexistent")
	if store.Len() != 2 {
		t.Fatalf("expected 2 after no-op delete, got %d", store.Len())
	}

	// Remaining entries should still be searchable
	matches, err := store.Search(context.Background(), "performance signal", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matches) != 2 {
		t.Fatalf("expected 2 matches, got %d", len(matches))
	}
}

func TestEmptyStore(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	matches, err := store.Search(context.Background(), "anything", 5)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matches) != 0 {
		t.Errorf("expected 0 matches from empty store, got %d", len(matches))
	}
}

func TestLimitGreaterThanCount(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	store.Add(
		Entry{ID: "a", Vector: []float32{1, 0, 0}},
		Entry{ID: "b", Vector: []float32{0, 1, 0}},
	)

	matches, err := store.Search(context.Background(), "performance signal", 100)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matches) != 2 {
		t.Errorf("expected 2 matches (limit>count), got %d", len(matches))
	}
}

func TestCosineSimilarity_Math(t *testing.T) {
	// Identical vectors → 1.0
	sim := cosineSimilarity([]float32{1, 0, 0}, []float32{1, 0, 0})
	if math.Abs(sim-1.0) > 1e-6 {
		t.Errorf("identical vectors: expected 1.0, got %f", sim)
	}

	// Orthogonal vectors → 0.0
	sim = cosineSimilarity([]float32{1, 0, 0}, []float32{0, 1, 0})
	if math.Abs(sim) > 1e-6 {
		t.Errorf("orthogonal vectors: expected 0.0, got %f", sim)
	}

	// Opposite vectors → -1.0
	sim = cosineSimilarity([]float32{1, 0, 0}, []float32{-1, 0, 0})
	if math.Abs(sim+1.0) > 1e-6 {
		t.Errorf("opposite vectors: expected -1.0, got %f", sim)
	}

	// 45 degree angle → ~0.707
	sim = cosineSimilarity([]float32{1, 0, 0}, []float32{1, 1, 0})
	expected := 1.0 / math.Sqrt(2.0)
	if math.Abs(sim-expected) > 1e-6 {
		t.Errorf("45deg: expected %f, got %f", expected, sim)
	}

	// Empty vectors → 0
	sim = cosineSimilarity([]float32{}, []float32{})
	if sim != 0 {
		t.Errorf("empty vectors: expected 0, got %f", sim)
	}

	// Mismatched lengths → 0
	sim = cosineSimilarity([]float32{1, 0}, []float32{1, 0, 0})
	if sim != 0 {
		t.Errorf("mismatched lengths: expected 0, got %f", sim)
	}

	// Zero vector → 0
	sim = cosineSimilarity([]float32{0, 0, 0}, []float32{1, 0, 0})
	if sim != 0 {
		t.Errorf("zero vector: expected 0, got %f", sim)
	}
}

func TestConcurrentAccess(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	// Pre-populate
	store.Add(
		Entry{ID: "a", Vector: []float32{1, 0, 0}},
		Entry{ID: "b", Vector: []float32{0, 1, 0}},
	)

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			store.Search(context.Background(), "performance signal", 5)
		}()
		go func(i int) {
			defer wg.Done()
			store.Add(Entry{
				ID:     "concurrent",
				Vector: []float32{float32(i), 0, 0},
			})
		}(i)
	}
	wg.Wait()
}

func TestMetadata(t *testing.T) {
	emb := newTestEmbedder()
	store := New(emb)

	err := store.AddTexts(context.Background(),
		[]string{"p1"},
		[]string{"performance signal"},
		[]map[string]string{{"source": "prometheus", "severity": "high"}},
	)
	if err != nil {
		t.Fatalf("AddTexts failed: %v", err)
	}

	matches, err := store.Search(context.Background(), "performance signal", 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(matches))
	}
	if matches[0].Metadata["source"] != "prometheus" {
		t.Errorf("expected metadata source=prometheus, got %v", matches[0].Metadata)
	}
}
