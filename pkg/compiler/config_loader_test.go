package compiler

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/MostViableProduct/great-wave/pkg/classifier"
	"github.com/MostViableProduct/great-wave/pkg/gate"
	"github.com/MostViableProduct/great-wave/pkg/health"
)

func TestLoadConfigFromFile_JSON(t *testing.T) {
	content := `{
		"classifier": {
			"categories": [
				{"name": "performance", "keywords": ["latency", "cpu"]}
			]
		},
		"gate": {
			"agreement_threshold": 0.85,
			"max_entries": 5000
		},
		"health": {
			"severities": [
				{"name": "critical", "weight": 0.6, "direction": "negative", "default_alpha": 5.0, "default_beta": 1.0},
				{"name": "improvement", "weight": 0.4, "direction": "positive", "default_alpha": 1.0, "default_beta": 3.0}
			]
		},
		"keywords": {
			"min_confidence": 0.8,
			"min_observations": 20
		},
		"max_vector_results": 10,
		"llm_min_confidence": 0.5
	}`

	path := filepath.Join(t.TempDir(), "config.json")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfigFromFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFromFile: %v", err)
	}

	if len(cfg.Classifier.Categories) != 1 {
		t.Fatalf("expected 1 category, got %d", len(cfg.Classifier.Categories))
	}
	if cfg.Classifier.Categories[0].Name != "performance" {
		t.Errorf("category name = %q, want %q", cfg.Classifier.Categories[0].Name, "performance")
	}
	if cfg.Gate.AgreementThreshold != 0.85 {
		t.Errorf("agreement_threshold = %v, want 0.85", cfg.Gate.AgreementThreshold)
	}
	if cfg.Gate.MaxEntries != 5000 {
		t.Errorf("max_entries = %d, want 5000", cfg.Gate.MaxEntries)
	}
	if len(cfg.Health.Severities) != 2 {
		t.Fatalf("expected 2 severities, got %d", len(cfg.Health.Severities))
	}
	if cfg.Health.Severities[0].DefaultAlpha != 5.0 {
		t.Errorf("default_alpha = %v, want 5.0", cfg.Health.Severities[0].DefaultAlpha)
	}
	if cfg.Health.Severities[0].DefaultBeta != 1.0 {
		t.Errorf("default_beta = %v, want 1.0", cfg.Health.Severities[0].DefaultBeta)
	}
	if cfg.Keywords.MinConfidence != 0.8 {
		t.Errorf("min_confidence = %v, want 0.8", cfg.Keywords.MinConfidence)
	}
	if cfg.Keywords.MinObservations != 20 {
		t.Errorf("min_observations = %d, want 20", cfg.Keywords.MinObservations)
	}
	if cfg.MaxVectorResults != 10 {
		t.Errorf("max_vector_results = %d, want 10", cfg.MaxVectorResults)
	}
	if cfg.LLMMinConfidence != 0.5 {
		t.Errorf("llm_min_confidence = %v, want 0.5", cfg.LLMMinConfidence)
	}
}

func TestLoadConfigFromFile_YAML(t *testing.T) {
	content := `
classifier:
  categories:
    - name: reliability
      keywords: [error, crash, panic]
      weights:
        crash: 2.0
gate:
  agreement_threshold: 0.90
  shadow_mode: true
health:
  severities:
    - name: critical
      weight: 0.5
      direction: negative
      default_alpha: 5.0
      default_beta: 1.0
    - name: improvement
      weight: 0.5
      direction: positive
      default_alpha: 1.0
      default_beta: 3.0
keywords:
  min_confidence: 0.6
max_vector_results: 3
llm_min_confidence: 0.4
`

	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfigFromFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFromFile: %v", err)
	}

	if cfg.Classifier.Categories[0].Name != "reliability" {
		t.Errorf("category name = %q, want %q", cfg.Classifier.Categories[0].Name, "reliability")
	}
	if cfg.Classifier.Categories[0].Weights["crash"] != 2.0 {
		t.Errorf("crash weight = %v, want 2.0", cfg.Classifier.Categories[0].Weights["crash"])
	}
	if cfg.Gate.AgreementThreshold != 0.90 {
		t.Errorf("agreement_threshold = %v, want 0.90", cfg.Gate.AgreementThreshold)
	}
	if !cfg.Gate.ShadowMode {
		t.Error("shadow_mode should be true")
	}
	if cfg.Health.Severities[0].DefaultAlpha != 5.0 {
		t.Errorf("default_alpha = %v, want 5.0", cfg.Health.Severities[0].DefaultAlpha)
	}
	if cfg.Keywords.MinConfidence != 0.6 {
		t.Errorf("min_confidence = %v, want 0.6", cfg.Keywords.MinConfidence)
	}
	if cfg.MaxVectorResults != 3 {
		t.Errorf("max_vector_results = %d, want 3", cfg.MaxVectorResults)
	}
}

func TestLoadConfigFromFile_YMLExtension(t *testing.T) {
	content := `
classifier:
  categories:
    - name: test
      keywords: [a, b]
`
	path := filepath.Join(t.TempDir(), "config.yml")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadConfigFromFile(path)
	if err != nil {
		t.Fatalf("LoadConfigFromFile with .yml: %v", err)
	}
	if cfg.Classifier.Categories[0].Name != "test" {
		t.Errorf("category name = %q, want %q", cfg.Classifier.Categories[0].Name, "test")
	}
}

func TestLoadConfigFromFile_UnsupportedExtension(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.toml")
	if err := os.WriteFile(path, []byte(""), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadConfigFromFile(path)
	if err == nil {
		t.Fatal("expected error for unsupported extension")
	}
}

func TestLoadConfigFromFile_FileNotFound(t *testing.T) {
	_, err := LoadConfigFromFile("/nonexistent/config.json")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadConfigFromFile_InvalidJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.json")
	if err := os.WriteFile(path, []byte("{not valid json"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadConfigFromFile(path)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestLoadConfigFromFile_InvalidYAML(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(path, []byte(":\n  :\n    - [invalid"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadConfigFromFile(path)
	if err == nil {
		t.Fatal("expected error for invalid YAML")
	}
}

func TestLoadConfigFromFile_ExampleConfigs(t *testing.T) {
	// Test that the example config files load without error
	examples := []string{
		"../../config/examples/observability.json",
		"../../config/examples/observability.yaml",
		"../../config/examples/product-health.json",
	}

	for _, path := range examples {
		t.Run(filepath.Base(path), func(t *testing.T) {
			cfg, err := LoadConfigFromFile(path)
			if err != nil {
				t.Fatalf("LoadConfigFromFile(%s): %v", path, err)
			}
			if len(cfg.Classifier.Categories) == 0 {
				t.Errorf("expected categories in %s", path)
			}
		})
	}
}

func TestValidateConfig_Valid(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "performance", Keywords: []string{"latency"}},
				{Name: "reliability", Keywords: []string{"error"}},
			},
		},
		Health: health.Config{
			Severities: []health.SeverityLevel{
				{Name: "critical", Weight: 0.6, Direction: "negative", DefaultAlpha: 5.0, DefaultBeta: 1.0},
				{Name: "improvement", Weight: 0.4, Direction: "positive", DefaultAlpha: 1.0, DefaultBeta: 3.0},
			},
		},
	}

	if err := ValidateConfig(cfg); err != nil {
		t.Errorf("expected valid config, got: %v", err)
	}
}

func TestValidateConfig_NoCategories(t *testing.T) {
	cfg := Config{}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for no categories")
	}
}

func TestValidateConfig_EmptyCategoryName(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "", Keywords: []string{"test"}},
			},
		},
	}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for empty category name")
	}
}

func TestValidateConfig_DuplicateCategory(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "perf", Keywords: []string{"a"}},
				{Name: "perf", Keywords: []string{"b"}},
			},
		},
	}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for duplicate category")
	}
}

func TestValidateConfig_InvalidDirection(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "test", Keywords: []string{"a"}},
			},
		},
		Health: health.Config{
			Severities: []health.SeverityLevel{
				{Name: "bad", Weight: 1.0, Direction: "sideways", DefaultAlpha: 1.0, DefaultBeta: 1.0},
			},
		},
	}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for invalid direction")
	}
}

func TestValidateConfig_WeightsSumWrong(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "test", Keywords: []string{"a"}},
			},
		},
		Health: health.Config{
			Severities: []health.SeverityLevel{
				{Name: "a", Weight: 0.3, Direction: "negative", DefaultAlpha: 1.0, DefaultBeta: 1.0},
				{Name: "b", Weight: 0.3, Direction: "positive", DefaultAlpha: 1.0, DefaultBeta: 1.0},
			},
		},
	}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for weights not summing to 1.0")
	}
}

func TestValidateConfig_InvalidAlpha(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "test", Keywords: []string{"a"}},
			},
		},
		Health: health.Config{
			Severities: []health.SeverityLevel{
				{Name: "a", Weight: 1.0, Direction: "negative", DefaultAlpha: 0, DefaultBeta: 1.0},
			},
		},
	}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for zero default_alpha")
	}
}

func TestValidateConfig_ConfidenceOutOfRange(t *testing.T) {
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "test", Keywords: []string{"a"}},
			},
		},
		Gate: gate.Config{AgreementThreshold: 1.5},
	}
	err := ValidateConfig(cfg)
	if err == nil {
		t.Fatal("expected error for agreement_threshold > 1")
	}
}

func TestValidateConfig_NoSeverities(t *testing.T) {
	// Config with no severities is valid (uses defaults at runtime)
	cfg := Config{
		Classifier: classifier.Config{
			Categories: []classifier.CategoryConfig{
				{Name: "test", Keywords: []string{"a"}},
			},
		},
	}
	if err := ValidateConfig(cfg); err != nil {
		t.Errorf("config with no severities should be valid: %v", err)
	}
}

func TestLoadConfigFromFile_JSONYAMLParity(t *testing.T) {
	jsonContent := `{
		"classifier": {
			"categories": [
				{"name": "perf", "keywords": ["latency"]}
			]
		},
		"gate": {"agreement_threshold": 0.85},
		"health": {
			"severities": [
				{"name": "critical", "weight": 1.0, "direction": "negative", "default_alpha": 5.0, "default_beta": 1.0}
			]
		},
		"max_vector_results": 7
	}`

	yamlContent := `
classifier:
  categories:
    - name: perf
      keywords: [latency]
gate:
  agreement_threshold: 0.85
health:
  severities:
    - name: critical
      weight: 1.0
      direction: negative
      default_alpha: 5.0
      default_beta: 1.0
max_vector_results: 7
`

	dir := t.TempDir()
	jsonPath := filepath.Join(dir, "config.json")
	yamlPath := filepath.Join(dir, "config.yaml")

	if err := os.WriteFile(jsonPath, []byte(jsonContent), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(yamlPath, []byte(yamlContent), 0644); err != nil {
		t.Fatal(err)
	}

	jsonCfg, err := LoadConfigFromFile(jsonPath)
	if err != nil {
		t.Fatalf("JSON load: %v", err)
	}
	yamlCfg, err := LoadConfigFromFile(yamlPath)
	if err != nil {
		t.Fatalf("YAML load: %v", err)
	}

	if jsonCfg.Classifier.Categories[0].Name != yamlCfg.Classifier.Categories[0].Name {
		t.Error("category name mismatch between JSON and YAML")
	}
	if jsonCfg.Gate.AgreementThreshold != yamlCfg.Gate.AgreementThreshold {
		t.Error("agreement_threshold mismatch between JSON and YAML")
	}
	if jsonCfg.Health.Severities[0].DefaultAlpha != yamlCfg.Health.Severities[0].DefaultAlpha {
		t.Error("default_alpha mismatch between JSON and YAML")
	}
	if jsonCfg.MaxVectorResults != yamlCfg.MaxVectorResults {
		t.Error("max_vector_results mismatch between JSON and YAML")
	}
}

func TestLoadConfigFromFile_ExampleConfigParity(t *testing.T) {
	jsonCfg, err := LoadConfigFromFile("../../config/examples/observability.json")
	if err != nil {
		t.Fatalf("JSON load: %v", err)
	}
	yamlCfg, err := LoadConfigFromFile("../../config/examples/observability.yaml")
	if err != nil {
		t.Fatalf("YAML load: %v", err)
	}

	// Category count and names
	if len(jsonCfg.Classifier.Categories) != len(yamlCfg.Classifier.Categories) {
		t.Fatalf("category count: json=%d yaml=%d",
			len(jsonCfg.Classifier.Categories), len(yamlCfg.Classifier.Categories))
	}
	for i, jc := range jsonCfg.Classifier.Categories {
		yc := yamlCfg.Classifier.Categories[i]
		if jc.Name != yc.Name {
			t.Errorf("category[%d] name: json=%q yaml=%q", i, jc.Name, yc.Name)
		}
	}

	// Gate config
	if jsonCfg.Gate.AgreementThreshold != yamlCfg.Gate.AgreementThreshold {
		t.Errorf("gate.agreement_threshold: json=%v yaml=%v",
			jsonCfg.Gate.AgreementThreshold, yamlCfg.Gate.AgreementThreshold)
	}

	// Health severities
	if len(jsonCfg.Health.Severities) != len(yamlCfg.Health.Severities) {
		t.Fatalf("severity count: json=%d yaml=%d",
			len(jsonCfg.Health.Severities), len(yamlCfg.Health.Severities))
	}
	for i, js := range jsonCfg.Health.Severities {
		ys := yamlCfg.Health.Severities[i]
		if js.Name != ys.Name || js.Weight != ys.Weight || js.DefaultAlpha != ys.DefaultAlpha {
			t.Errorf("severity[%d] mismatch: json=%+v yaml=%+v", i, js, ys)
		}
	}

	// Top-level
	if jsonCfg.MaxVectorResults != yamlCfg.MaxVectorResults {
		t.Errorf("max_vector_results: json=%d yaml=%d",
			jsonCfg.MaxVectorResults, yamlCfg.MaxVectorResults)
	}
	if jsonCfg.LLMMinConfidence != yamlCfg.LLMMinConfidence {
		t.Errorf("llm_min_confidence: json=%v yaml=%v",
			jsonCfg.LLMMinConfidence, yamlCfg.LLMMinConfidence)
	}
}

