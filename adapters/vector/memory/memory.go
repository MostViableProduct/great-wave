// Package memoryvec provides an in-memory vector store implementing
// compiler.VectorSearcher with cosine similarity search.
//
// Usage:
//
//	store := memoryvec.New(embedder)
//	store.AddTexts(ctx, []string{"id1"}, []string{"some text"}, nil)
//	matches, _ := store.Search(ctx, "query", 5)
package memoryvec

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

// Compile-time interface check.
var _ compiler.VectorSearcher = (*Store)(nil)

// Entry is a single vector entry with its ID and metadata.
type Entry struct {
	ID       string
	Vector   []float32
	Metadata map[string]string
}

// Store is a thread-safe in-memory vector store with cosine similarity search.
type Store struct {
	mu       sync.RWMutex
	entries  []Entry
	index    map[string]int // id → index in entries
	embedder compiler.Embedder
}

// New creates a Store backed by the given embedder.
func New(embedder compiler.Embedder) *Store {
	return &Store{
		index:    make(map[string]int),
		embedder: embedder,
	}
}

// Add inserts pre-embedded entries. If an entry with the same ID exists,
// it is replaced.
func (s *Store) Add(entries ...Entry) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, e := range entries {
		if idx, ok := s.index[e.ID]; ok {
			s.entries[idx] = e
		} else {
			s.index[e.ID] = len(s.entries)
			s.entries = append(s.entries, e)
		}
	}
}

// AddTexts embeds the given texts and adds them as entries. IDs and texts
// must have the same length. Metadata is optional (may be nil or shorter).
func (s *Store) AddTexts(ctx context.Context, ids, texts []string, metadata []map[string]string) error {
	if len(ids) != len(texts) {
		return fmt.Errorf("memoryvec: ids and texts must have the same length (got %d ids, %d texts)", len(ids), len(texts))
	}
	vectors, err := s.embedder.Embed(ctx, texts)
	if err != nil {
		return err
	}
	if len(vectors) != len(ids) {
		return fmt.Errorf("memoryvec: embedder returned %d vectors for %d inputs", len(vectors), len(ids))
	}

	entries := make([]Entry, len(ids))
	for i := range ids {
		var meta map[string]string
		if i < len(metadata) {
			meta = metadata[i]
		}
		entries[i] = Entry{
			ID:       ids[i],
			Vector:   vectors[i],
			Metadata: meta,
		}
	}

	s.Add(entries...)
	return nil
}

// Delete removes entries by ID.
func (s *Store) Delete(ids ...string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		if idx, ok := s.index[id]; ok {
			// Swap with last element for O(1) removal
			last := len(s.entries) - 1
			if idx != last {
				s.entries[idx] = s.entries[last]
				s.index[s.entries[idx].ID] = idx
			}
			s.entries = s.entries[:last]
			delete(s.index, id)
		}
	}
}

// Len returns the number of entries in the store.
func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.entries)
}

// Search implements compiler.VectorSearcher. It embeds the query, computes
// cosine similarity against all stored vectors, and returns the top-k matches.
func (s *Store) Search(ctx context.Context, query string, limit int) ([]compiler.VectorMatch, error) {
	if limit <= 0 {
		return nil, nil
	}
	vectors, err := s.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	if len(vectors) == 0 || len(vectors[0]) == 0 {
		return nil, nil
	}
	queryVec := vectors[0]

	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.entries) == 0 {
		return nil, nil
	}

	type scored struct {
		entry Entry
		score float64
	}
	results := make([]scored, 0, len(s.entries))
	for _, e := range s.entries {
		sim := cosineSimilarity(queryVec, e.Vector)
		results = append(results, scored{entry: e, score: sim})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if limit > len(results) {
		limit = len(results)
	}

	matches := make([]compiler.VectorMatch, limit)
	for i := 0; i < limit; i++ {
		matches[i] = compiler.VectorMatch{
			ID:       results[i].entry.ID,
			Score:    results[i].score,
			Metadata: results[i].entry.Metadata,
		}
	}

	return matches, nil
}

// cosineSimilarity computes the cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
