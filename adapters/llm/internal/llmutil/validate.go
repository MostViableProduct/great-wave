// Package llmutil provides shared validation helpers for LLM adapter responses.
package llmutil

import (
	"fmt"
	"math"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

// ValidateResult validates an LLM classification result for confidence range,
// keyword count, and category membership. adapterName is used in error messages.
// Keywords are capped in-place if they exceed 50 entries.
func ValidateResult(result *compiler.LLMResult, categories []string, adapterName string) error {
	if math.IsNaN(result.Confidence) || math.IsInf(result.Confidence, 0) || result.Confidence < 0 || result.Confidence > 1 {
		return fmt.Errorf("%s: confidence %f out of range [0,1]", adapterName, result.Confidence)
	}
	if len(result.Keywords) > 50 {
		result.Keywords = result.Keywords[:50]
	}
	for _, cat := range categories {
		if cat == result.Category {
			return nil
		}
	}
	return fmt.Errorf("%s: LLM returned unknown category %q", adapterName, result.Category)
}
