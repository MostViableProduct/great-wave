// Example product-health demonstrates health scoring with event recording
// and keyword promotion.
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

func main() {
	cfg := compiler.DefaultConfig()
	cfg.Classifier = classifier.Config{
		Categories: []classifier.CategoryConfig{
			{Name: "performance", Keywords: []string{"latency", "p99", "slow"}},
			{Name: "reliability", Keywords: []string{"error", "crash", "outage"}},
			{Name: "deployment", Keywords: []string{"deploy", "rollback", "release"}},
		},
		TypeToCategory: map[string]string{
			"metric": "performance",
			"error":  "reliability",
			"deploy": "deployment",
		},
	}

	callbacks := compiler.Callbacks{
		OnKeywordsPromoted: func(count int) {
			fmt.Printf("Promoted %d keywords\n", count)
		},
	}

	c := compiler.New(cfg, compiler.Deps{}, callbacks)

	// Simulate a series of events affecting entity health
	events := []struct {
		severity   string
		category   string
		confidence float64
	}{
		{"critical", "reliability", 1.0},
		{"critical", "reliability", 0.9},
		{"warning", "performance", 0.8},
		{"improvement", "deployment", 0.7},
		{"improvement", "deployment", 0.9},
	}

	fmt.Println("Recording health events for api-service...")
	for _, e := range events {
		c.RecordHealthEvent("prod", "api-service", e.severity, e.category, e.confidence)
		health := c.ScoreHealth("prod", "api-service")
		fmt.Printf("  After %s/%s: score=%.1f CI=[%.1f, %.1f]\n",
			e.severity, e.category, health.Score,
			health.ConfidenceIntervalLower, health.ConfidenceIntervalUpper)
	}

	// Classify and show related entity resolution
	result, err := c.Classify(context.Background(), compiler.Signal{
		Source:  "prometheus",
		Type:    "metric",
		Content: "p99 latency spike in api-service",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nClassification: category=%s score=%.2f\n", result.Category, result.RelevanceScore)

	// Run keyword promotion (no learned keywords yet, so 0 promoted)
	count, _ := c.PromoteKeywords()
	fmt.Printf("Keywords promoted: %d\n", count)
}
