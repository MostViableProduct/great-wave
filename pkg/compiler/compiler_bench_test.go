package compiler

import (
	"context"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
)

func BenchmarkCompilerClassify_HeuristicOnly(b *testing.B) {
	cfg := DefaultConfig()
	cfg.Classifier = classifier.Config{
		Categories: []classifier.CategoryConfig{
			{Name: "performance", Keywords: []string{"latency", "throughput", "p50", "p95", "p99", "slow", "timeout"}, Weights: map[string]float64{"p99": 2.0}},
			{Name: "reliability", Keywords: []string{"error", "failure", "crash", "panic", "exception", "outage"}, Weights: map[string]float64{"outage": 2.0}},
			{Name: "deployment", Keywords: []string{"deploy", "release", "rollout", "rollback", "build"}, Weights: map[string]float64{"deploy": 2.0}},
		},
		TypeToCategory: map[string]string{"metric": "performance", "error": "reliability"},
	}
	c := New(cfg, Deps{}, Callbacks{})
	ctx := context.Background()
	sig := Signal{Source: "test", Type: "metric", Content: "High p99 latency detected in API gateway"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Classify(ctx, sig)
	}
}
