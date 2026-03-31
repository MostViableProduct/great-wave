// Package keywords implements self-improving keyword learning for the cascade
// classifier. When the heuristic and LLM classifiers disagree, novel keywords
// are extracted and tracked. Keywords that accumulate sufficient confidence are
// promoted to the heuristic classifier's runtime store.
package keywords

import (
	"strings"
	"unicode"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
)

// KeywordStore defines the persistence interface for learned keywords.
type KeywordStore interface {
	UpsertKeyword(keyword, category string) error
	PromoteKeywords(minConfidence float64, minObservations int) ([]classifier.LearnedKeyword, error)
	DemoteKeywords(minConfidence float64, minObservations int) error
	LoadPromotedKeywords(limit int) ([]classifier.LearnedKeyword, error)
	WeakenKeyword(keyword string) error
}

// Config holds keyword learning configuration.
type Config struct {
	// MinConfidence is the minimum confidence to promote a keyword (default: 0.7).
	MinConfidence float64 `json:"min_confidence" yaml:"min_confidence"`
	// MinObservations is the minimum observations to promote (default: 10).
	MinObservations int `json:"min_observations" yaml:"min_observations"`
	// MaxPromoted is the maximum promoted keywords to load (default: 200).
	MaxPromoted int `json:"max_promoted" yaml:"max_promoted"`
	// MaxExtracted is the maximum keywords to extract per classification (default: 5).
	MaxExtracted int `json:"max_extracted" yaml:"max_extracted"`
}

// DefaultConfig returns default keyword learning configuration.
func DefaultConfig() Config {
	return Config{
		MinConfidence:   0.7,
		MinObservations: 10,
		MaxPromoted:     200,
		MaxExtracted:    5,
	}
}

// Learner manages keyword learning and promotion.
type Learner struct {
	config         Config
	store          KeywordStore
	classifier     *classifier.Classifier
	runtime        *classifier.LearnedKeywordStore
	staticKeywords map[string]bool // pre-computed at construction, immutable
}

// NewLearner creates a new keyword learner.
func NewLearner(cfg Config, store KeywordStore, cls *classifier.Classifier, runtime *classifier.LearnedKeywordStore) *Learner {
	if cfg.MinConfidence == 0 {
		cfg.MinConfidence = 0.7
	}
	if cfg.MinObservations == 0 {
		cfg.MinObservations = 10
	}
	if cfg.MaxPromoted == 0 {
		cfg.MaxPromoted = 200
	}
	if cfg.MaxExtracted == 0 {
		cfg.MaxExtracted = 5
	}

	static := make(map[string]bool)
	for _, kws := range cls.CategoryKeywords() {
		for _, kw := range kws {
			static[kw] = true
		}
	}

	return &Learner{
		config:         cfg,
		store:          store,
		classifier:     cls,
		runtime:        runtime,
		staticKeywords: static,
	}
}

// RecordDisagreement records when heuristic and LLM disagree, extracting
// novel keywords and upserting them for learning.
func (l *Learner) RecordDisagreement(content string, llmCategory string, llmConfidence float64) {
	if l.store == nil || llmConfidence < l.config.MinConfidence {
		return
	}

	keywords := l.ExtractKeywords(content, llmCategory)
	for _, kw := range keywords {
		_ = l.store.UpsertKeyword(kw, llmCategory)
	}
}

// ExtractKeywords tokenizes content and returns up to MaxExtracted novel
// tokens not already in the static category keywords map.
func (l *Learner) ExtractKeywords(content, category string) []string {
	tokens := Tokenize(content)

	seen := make(map[string]bool)
	var novel []string
	for _, token := range tokens {
		if len(token) < 3 {
			continue
		}
		if Stopwords[token] {
			continue
		}
		if l.staticKeywords[token] {
			continue
		}
		if seen[token] {
			continue
		}
		seen[token] = true
		novel = append(novel, token)
		if len(novel) >= l.config.MaxExtracted {
			break
		}
	}

	return novel
}

// Promote queries the store for high-confidence keywords and updates
// the runtime store. Keywords below threshold are automatically demoted.
func (l *Learner) Promote() (int, error) {
	if l.store == nil {
		return 0, nil
	}

	keywords, err := l.store.PromoteKeywords(l.config.MinConfidence, l.config.MinObservations)
	if err != nil {
		return 0, err
	}

	if err := l.store.DemoteKeywords(l.config.MinConfidence, l.config.MinObservations); err != nil {
		return 0, err
	}

	l.runtime.Update(keywords)
	return len(keywords), nil
}

// LoadPromoted loads previously promoted keywords from the store into the runtime.
func (l *Learner) LoadPromoted() (int, error) {
	if l.store == nil {
		return 0, nil
	}

	keywords, err := l.store.LoadPromotedKeywords(l.config.MaxPromoted)
	if err != nil {
		return 0, err
	}

	l.runtime.Update(keywords)
	return len(keywords), nil
}

// WeakenKeyword decrements the observation count for a keyword, recording a
// false-positive signal that makes the keyword more likely to be demoted.
func (l *Learner) WeakenKeyword(keyword string) error {
	if l.store == nil {
		return nil
	}
	return l.store.WeakenKeyword(keyword)
}

// Tokenize splits content into lowercase tokens on whitespace and punctuation.
func Tokenize(content string) []string {
	lower := strings.ToLower(content)
	return strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_'
	})
}

// Stopwords that should never become learned keywords.
var Stopwords = map[string]bool{
	"the": true, "and": true, "for": true, "with": true, "this": true,
	"that": true, "from": true, "are": true, "was": true, "were": true,
	"been": true, "have": true, "has": true, "had": true, "not": true,
	"but": true, "all": true, "can": true, "her": true, "his": true,
	"its": true, "may": true, "new": true, "now": true, "old": true,
	"see": true, "way": true, "who": true, "did": true, "get": true,
	"let": true, "say": true, "she": true, "too": true, "use": true,
}
