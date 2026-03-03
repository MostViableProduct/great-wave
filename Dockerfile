# For production, pin by digest: golang:1.25-alpine@sha256:<digest>
FROM golang:1.25-alpine AS builder

RUN apk add --no-cache ca-certificates

WORKDIR /build

COPY go.mod go.sum* ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /compiler ./cmd/compiler/
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /healthcheck ./cmd/healthcheck/

# For production, pin by digest: gcr.io/distroless/static-debian12:nonroot@sha256:<digest>
FROM gcr.io/distroless/static-debian12:nonroot

COPY --from=builder /compiler /compiler
COPY --from=builder /healthcheck /healthcheck
COPY --from=builder /build/config/examples/ /config/examples/

EXPOSE 8200

USER nonroot:nonroot

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD ["/healthcheck"]

ENTRYPOINT ["/compiler"]
