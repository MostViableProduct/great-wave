// Package openai implements compiler.LLMClassifier using OpenAI's Chat Completions API.
//
// The adapter sends classification requests with structured JSON output and
// validates that the returned category is in the provided list.
//
// Usage:
//
//	client := openai.New(os.Getenv("OPENAI_API_KEY"))
//	result, err := client.Classify(ctx, "high p99 latency", "metric", []string{"performance", "reliability"})
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/MostViableProduct/great-wave/adapters/llm/internal/llmutil"
	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

const (
	defaultModel   = "gpt-4o"
	defaultBaseURL = "https://api.openai.com"
	defaultTimeout = 30 * time.Second
	maxTokens      = 256
)

// Compile-time interface check.
var _ compiler.LLMClassifier = (*Client)(nil)

// Client calls OpenAI's Chat Completions API for classification.
type Client struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
}

// Option configures a Client.
type Option func(*Client)

// modelNameRe restricts model names to safe characters.
var modelNameRe = regexp.MustCompile(`^[a-zA-Z0-9._-]+$`)

// WithModel overrides the default model.
func WithModel(model string) Option {
	return func(c *Client) {
		if modelNameRe.MatchString(model) {
			c.model = model
		}
	}
}

// WithTimeout overrides the default HTTP timeout.
func WithTimeout(d time.Duration) Option {
	return func(c *Client) { c.httpClient.Timeout = d }
}

// WithBaseURL overrides the API base URL (useful for proxies or testing).
func WithBaseURL(u string) Option {
	return func(c *Client) {
		u = strings.TrimRight(u, "/")
		parsed, err := url.Parse(u)
		if err != nil || (parsed.Scheme != "https" && parsed.Scheme != "http") {
			return // silently ignore invalid URLs
		}
		c.baseURL = u
	}
}

// New creates an OpenAI classification client.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		httpClient: &http.Client{
			Timeout: defaultTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		apiKey:  apiKey,
		model:   defaultModel,
		baseURL: defaultBaseURL,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

const systemPrompt = `You are a signal classifier. Given a signal's content and type, classify it into exactly one of the provided categories.

Respond with ONLY a JSON object in this exact format:
{"category":"<one of the provided categories>","confidence":<0.0 to 1.0>,"keywords":["<relevant keywords>"]}

Rules:
- category MUST be one of the provided categories, exactly as written
- confidence is your certainty from 0.0 (no idea) to 1.0 (certain)
- keywords are the terms from the content that drove your decision`

// Classify implements compiler.LLMClassifier.
func (c *Client) Classify(ctx context.Context, content, signalType string, categories []string) (*compiler.LLMResult, error) {
	userPrompt := fmt.Sprintf("Categories: %s\nSignal type: %s\nContent: %s",
		strings.Join(categories, ", "), signalType, content)

	reqBody := map[string]any{
		"model": c.model,
		"messages": []map[string]string{
			{"role": "system", "content": systemPrompt},
			{"role": "user", "content": userPrompt},
		},
		"max_tokens":      maxTokens,
		"response_format": map[string]string{"type": "json_object"},
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req) //#nosec G704 -- URL is admin-configured API endpoint
	if err != nil {
		return nil, fmt.Errorf("openai: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		log.Printf("llm: request failed (status %d): %s", resp.StatusCode, string(respBody))
		return nil, fmt.Errorf("llm: request failed with status %d", resp.StatusCode)
	}

	var chatResp struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("openai: decode response: %w", err)
	}
	if len(chatResp.Choices) == 0 {
		return nil, fmt.Errorf("openai: no choices in response")
	}

	var result compiler.LLMResult
	if err := json.Unmarshal([]byte(chatResp.Choices[0].Message.Content), &result); err != nil {
		return nil, fmt.Errorf("openai: parse classification JSON: %w", err)
	}

	if err := llmutil.ValidateResult(&result, categories, "openai"); err != nil {
		return nil, err
	}

	return &result, nil
}
