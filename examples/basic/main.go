// Example basic demonstrates minimal classification with in-memory dependencies.
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
			{Name: "performance", Keywords: []string{"latency", "p99", "slow", "timeout"}},
			{Name: "reliability", Keywords: []string{"error", "crash", "outage", "failure"}},
		},
		TypeToCategory: map[string]string{
			"metric": "performance",
			"error":  "reliability",
		},
	}

	c := compiler.New(cfg, compiler.Deps{}, compiler.Callbacks{})

	signals := []compiler.Signal{
		{Source: "prometheus", Type: "metric", Content: "p99 latency spike to 450ms"},
		{Source: "sentry", Type: "error", Content: "database connection pool exhausted"},
		{Source: "grafana", Type: "alert", Content: "CPU usage at 95%"},
	}

	for _, sig := range signals {
		result, err := c.Classify(context.Background(), sig)
		if err != nil {
			log.Printf("Error: %v", err)
			continue
		}
		fmt.Printf("%-12s → category=%-15s source=%-16s score=%.2f\n",
			sig.Type, result.Category, result.ClassificationSource, result.RelevanceScore)
	}
}
