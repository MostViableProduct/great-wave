package compiler

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"
)

// LoadConfigFromFile loads a Config from a JSON or YAML file.
// The format is auto-detected by file extension (.json, .yaml, .yml).
func LoadConfigFromFile(path string) (Config, error) {
	data, err := os.ReadFile(path) //#nosec G304 -- path is admin-controlled via CONFIG_PATH env var
	if err != nil {
		return Config{}, fmt.Errorf("read config file: %w", err)
	}

	ext := strings.ToLower(filepath.Ext(path))

	var cfg Config
	switch ext {
	case ".json":
		if err := json.Unmarshal(data, &cfg); err != nil {
			return Config{}, fmt.Errorf("parse JSON config: %w", err)
		}
	case ".yaml", ".yml":
		if err := yaml.Unmarshal(data, &cfg); err != nil {
			return Config{}, fmt.Errorf("parse YAML config: %w", err)
		}
	default:
		return Config{}, fmt.Errorf("unsupported config file extension %q (use .json, .yaml, or .yml)", ext)
	}

	return cfg, nil
}

// ValidateConfig checks that a Config is internally consistent.
func ValidateConfig(cfg Config) error {
	// At least one category defined
	if len(cfg.Classifier.Categories) == 0 {
		return fmt.Errorf("classifier: at least one category must be defined")
	}

	// Category names are non-empty and unique
	seen := make(map[string]bool, len(cfg.Classifier.Categories))
	for i, cat := range cfg.Classifier.Categories {
		name := strings.TrimSpace(cat.Name)
		if name == "" {
			return fmt.Errorf("classifier: category[%d] has an empty name", i)
		}
		if seen[name] {
			return fmt.Errorf("classifier: duplicate category name %q", name)
		}
		seen[name] = true
	}

	// Severity validation
	if len(cfg.Health.Severities) > 0 {
		weightSum := 0.0
		for i, sev := range cfg.Health.Severities {
			name := strings.TrimSpace(sev.Name)
			if name == "" {
				return fmt.Errorf("health: severity[%d] has an empty name", i)
			}
			if sev.Direction != "positive" && sev.Direction != "negative" {
				return fmt.Errorf("health: severity %q has invalid direction %q (must be \"positive\" or \"negative\")", name, sev.Direction)
			}
			if sev.Weight < 0 || sev.Weight > 1 {
				return fmt.Errorf("health: severity %q weight %.2f is outside [0, 1]", name, sev.Weight)
			}
			if sev.DefaultAlpha <= 0 {
				return fmt.Errorf("health: severity %q default_alpha must be > 0", name)
			}
			if sev.DefaultBeta <= 0 {
				return fmt.Errorf("health: severity %q default_beta must be > 0", name)
			}
			weightSum += sev.Weight
		}

		// Weights should sum to approximately 1.0 (tolerance: 0.05)
		if math.Abs(weightSum-1.0) > 0.05 {
			return fmt.Errorf("health: severity weights sum to %.4f, expected ~1.0 (tolerance 0.05)", weightSum)
		}
	}

	// Numeric threshold ranges
	if cfg.Gate.AgreementThreshold != 0 && (cfg.Gate.AgreementThreshold < 0 || cfg.Gate.AgreementThreshold > 1) {
		return fmt.Errorf("gate: agreement_threshold %.2f is outside [0, 1]", cfg.Gate.AgreementThreshold)
	}
	if cfg.Gate.UncertaintyMax != 0 && (cfg.Gate.UncertaintyMax < 0 || cfg.Gate.UncertaintyMax > 1) {
		return fmt.Errorf("gate: uncertainty_max %.2f is outside [0, 1]", cfg.Gate.UncertaintyMax)
	}
	if cfg.Gate.HeuristicConfidence != 0 && (cfg.Gate.HeuristicConfidence < 0 || cfg.Gate.HeuristicConfidence > 1) {
		return fmt.Errorf("gate: heuristic_confidence %.2f is outside [0, 1]", cfg.Gate.HeuristicConfidence)
	}
	if cfg.Keywords.MinConfidence != 0 && (cfg.Keywords.MinConfidence < 0 || cfg.Keywords.MinConfidence > 1) {
		return fmt.Errorf("keywords: min_confidence %.2f is outside [0, 1]", cfg.Keywords.MinConfidence)
	}
	if cfg.LLMMinConfidence != 0 && (cfg.LLMMinConfidence < 0 || cfg.LLMMinConfidence > 1) {
		return fmt.Errorf("llm_min_confidence %.2f is outside [0, 1]", cfg.LLMMinConfidence)
	}

	return nil
}
