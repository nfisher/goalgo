SHELL := /bin/bash -eu -o pipefail
SRC := $(shell find . -path ./vendor -prune -o -name '*.go')
COVERAGE_PROFILE := results/coverage.out
COVERAGE_HTML := results/coverage.html

.PHONY: all
all: install

.PHONY: install
install: test vet
	go install -v ./...

.PHONY: test
test: $(COVERAGE_PROFILE)

.PHONY: html
html: $(COVERAGE_HTML)

results:
	mkdir -p results

$(COVERAGE_PROFILE): results $(SRC)
	go test -v -coverprofile=$(COVERAGE_PROFILE) ./...

$(COVERAGE_HTML): $(COVERAGE_PROFILE)
	go tool cover -html=$(COVERAGE_PROFILE) -o $(COVERAGE_HTML)

.PHONY: vet
vet:
	go vet -all ./...
