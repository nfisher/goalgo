SHELL := /bin/bash -eu -o pipefail
SRC := $(shell find . -path ./vendor -prune -o -name '*.go')
COVERAGE_PROFILE := results/coverage.out
COVERAGE_HTML := results/coverage.html

.PHONY: all
all: install benchmark

.PHONY: install
install: get test vet
	go install -v ./...

.PHONY: get
get:
	go get -d -u gonum.org/v1/gonum/mat
	go get -d -u gonum.org/v1/gonum/blas/blas64

.PHONY: test
test: $(COVERAGE_PROFILE)

.PHONY: html
html: $(COVERAGE_HTML)

.PHONY: benchmark
benchmark:
	go test -short -benchmem -bench=. ./...

.PHONY: benchmarklong
benchmarklong:
	go test -benchmem -bench=. ./...

results:
	mkdir -p results

$(COVERAGE_PROFILE): results $(SRC)
	go test -v -coverprofile=$(COVERAGE_PROFILE) ./...

$(COVERAGE_HTML): $(COVERAGE_PROFILE)
	go tool cover -html=$(COVERAGE_PROFILE) -o $(COVERAGE_HTML)

.PHONY: vet
vet:
	go vet -all ./...
