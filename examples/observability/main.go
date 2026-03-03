// Example observability demonstrates full setup with LLM stub, events, and callbacks.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/MostViableProduct/great-wave/adapters/events/logwriter"
	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

func main() {
	cfg := compiler.DefaultConfig()
	cfg.Classifier = classifier.Config{
		Categories: []classifier.CategoryConfig{
			{Name: "performance", Keywords: []string{"latency", "p99", "slow", "timeout"}, Weights: map[string]float64{"p99": 2.0}},
			{Name: "reliability", Keywords: []string{"error", "crash", "outage"}, Weights: map[string]float64{"outage": 2.0}},
			{Name: "security", Keywords: []string{"auth", "vulnerability", "cve"}, Weights: map[string]float64{"cve": 2.0}},
		},
		SourcePriors: map[string]map[string]float64{
			"prometheus": {"performance": 3.0},
			"sentry":     {"reliability": 3.0},
		},
		TypeToCategory: map[string]string{
			"metric": "performance",
			"error":  "reliability",
		},
	}

	deps := compiler.Deps{
		Events: logwriter.New(os.Stdout),
	}

	callbacks := compiler.Callbacks{
		OnClassify: func(source, category string) {
			fmt.Printf("[callback] classify: source=%s category=%s\n", source, category)
		},
		OnGateSkip: func() {
			fmt.Println("[callback] gate: skipped LLM call")
		},
		OnLLMFallback: func(err error) {
			fmt.Printf("[callback] llm: fallback: %v\n", err)
		},
		OnKeywordsPromoted: func(count int) {
			fmt.Printf("[callback] keywords: promoted %d\n", count)
		},
	}

	c := compiler.New(cfg, deps, callbacks)

	// Classify with payload for signal class inference
	result, err := c.Classify(context.Background(), compiler.Signal{
		Source:  "prometheus",
		Type:    "metric",
		Content: "p99 latency spike to 450ms in API gateway",
		Payload: json.RawMessage(`{"value": 450, "unit": "ms"}`),
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("\nResult: category=%s source=%s class=%s score=%.2f\n",
		result.Category, result.ClassificationSource, result.SignalClass, result.RelevanceScore)

	// Health scoring
	c.RecordHealthEvent("t1", "api-gateway", "critical", "performance", 0.9)
	c.RecordHealthEvent("t1", "api-gateway", "warning", "performance", 0.7)

	health := c.ScoreHealth("t1", "api-gateway")
	fmt.Printf("\nHealth: entity=%s score=%.1f CI=[%.1f, %.1f]\n",
		health.EntityID, health.Score, health.ConfidenceIntervalLower, health.ConfidenceIntervalUpper)
}
