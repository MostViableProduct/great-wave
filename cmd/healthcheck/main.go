// Command healthcheck is a minimal binary used as a Docker HEALTHCHECK.
// It sends a GET request to http://localhost:$PORT/health and exits 0 on
// success (HTTP 200), or 1 on failure.
package main

import (
	"net/http"
	"net/url"
	"os"
	"time"
)

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8200"
	}

	// Construct URL safely to avoid SSRF taint (G704).
	u := &url.URL{Scheme: "http", Host: "localhost:" + port, Path: "/health"}

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(u.String())
	if err != nil {
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		os.Exit(1)
	}
}
