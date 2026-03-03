package classifier

import "sync"

// LearnedKeyword represents a keyword discovered through LLM classification
// that can augment the static heuristic classifier.
type LearnedKeyword struct {
	Keyword              string  `json:"keyword"`
	Category             string  `json:"category"`
	Weight               float64 `json:"weight"`
	Confidence           float64 `json:"confidence"`
	TotalObservations    int     `json:"total_observations"`
	PositiveObservations int     `json:"positive_observations"`
}

// LearnedKeywordStore is a thread-safe runtime cache of promoted learned keywords.
// Both GetPromotedKeywords and GetPromotedWeights return pre-built maps that are
// rebuilt only when Update is called, avoiding per-classify allocations.
type LearnedKeywordStore struct {
	mu             sync.RWMutex
	keywords       []LearnedKeyword
	cachedKeywords map[string][]string
	cachedWeights  map[string]float64
}

// NewLearnedKeywordStore creates an empty store.
func NewLearnedKeywordStore() *LearnedKeywordStore {
	return &LearnedKeywordStore{
		cachedKeywords: make(map[string][]string),
		cachedWeights:  make(map[string]float64),
	}
}

// GetPromotedKeywords returns the cached map of category -> keyword list.
// The returned map is shared and must not be modified by the caller.
func (s *LearnedKeywordStore) GetPromotedKeywords() map[string][]string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cachedKeywords
}

// GetPromotedWeights returns the cached map of keyword -> weight.
// The returned map is shared and must not be modified by the caller.
func (s *LearnedKeywordStore) GetPromotedWeights() map[string]float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cachedWeights
}

// Update replaces the entire set of promoted keywords atomically and
// rebuilds the cached lookup maps.
func (s *LearnedKeywordStore) Update(keywords []LearnedKeyword) {
	kw := make(map[string][]string)
	weights := make(map[string]float64, len(keywords))
	for _, k := range keywords {
		kw[k.Category] = append(kw[k.Category], k.Keyword)
		weights[k.Keyword] = k.Weight
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	s.keywords = keywords
	s.cachedKeywords = kw
	s.cachedWeights = weights
}

// Count returns the number of currently promoted keywords.
func (s *LearnedKeywordStore) Count() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.keywords)
}
