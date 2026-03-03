.PHONY: build test test-unit test-integration lint vet bench docker clean

build:
	go build -ldflags="-s -w" -o bin/compiler ./cmd/compiler/

test:
	go test -count=1 -race ./...

test-unit:
	go test -count=1 -race -short ./...

test-integration:
	go test -count=1 -race -run Integration ./...

lint: vet
	@echo "Lint passed"

vet:
	go vet ./...

bench:
	go test -bench=. -benchmem ./...

docker:
	docker build -t contextual-compiler .

clean:
	rm -rf bin/
