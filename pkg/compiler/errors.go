package compiler

import "errors"

var (
	// ErrNoLLM is returned when LLM classification is attempted but no LLM adapter is configured.
	ErrNoLLM = errors.New("no LLM classifier configured")

	// ErrEmptyContent is returned when a signal has no classifiable content.
	ErrEmptyContent = errors.New("signal content is empty")

	// ErrInvalidCategory is returned when an LLM returns a category not in the configured set.
	ErrInvalidCategory = errors.New("LLM returned invalid category")
)
