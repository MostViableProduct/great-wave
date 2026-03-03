// Package openaiembed implements compiler.Embedder using OpenAI's Embeddings API.
//
// Usage:
//
//	embedder := openaiembed.New(os.Getenv("OPENAI_API_KEY"))
//	vectors, err := embedder.Embed(ctx, []string{"high p99 latency"})
package openaiembed

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/MostViableProduct/great-wave/pkg/compiler"
)

const (
	defaultModel      = "text-embedding-3-small"
	defaultBaseURL    = "https://api.openai.com"
	defaultTimeout    = 30 * time.Second
	defaultDimensions = 1536
)

// Compile-time interface check.
var _ compiler.Embedder = (*Client)(nil)

// Client calls OpenAI's Embeddings API.
type Client struct {
	httpClient *http.Client
	apiKey     string
	model      string
	baseURL    string
	dimensions int
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

// WithDimensions overrides the default embedding dimensions.
func WithDimensions(d int) Option {
	return func(c *Client) { c.dimensions = d }
}

// New creates an OpenAI embedding client.
func New(apiKey string, opts ...Option) *Client {
	c := &Client{
		httpClient: &http.Client{
			Timeout: defaultTimeout,
			CheckRedirect: func(req *http.Request, via []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		apiKey:     apiKey,
		model:      defaultModel,
		baseURL:    defaultBaseURL,
		dimensions: defaultDimensions,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// Dimensions returns the embedding vector size.
func (c *Client) Dimensions() int { return c.dimensions }

// Embed implements compiler.Embedder.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	reqBody := map[string]any{
		"model":      c.model,
		"input":      texts,
		"dimensions": c.dimensions,
	}

	body, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("openai-embed: marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("openai-embed: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req) //#nosec G704 -- URL is admin-configured API endpoint
	if err != nil {
		return nil, fmt.Errorf("openai-embed: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("openai-embed: returned %d: %s", resp.StatusCode, string(respBody))
	}

	var embResp struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
			Index     int       `json:"index"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("openai-embed: decode response: %w", err)
	}

	// Sort by index to match input order
	result := make([][]float32, len(texts))
	for _, d := range embResp.Data {
		if d.Index < len(result) {
			result[d.Index] = d.Embedding
		}
	}

	return result, nil
}
