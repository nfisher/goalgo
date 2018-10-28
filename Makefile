SHELL := /bin/bash -eu -o pipefail

.PHONY: all
all: install

.PHONY: install
install: test vet
	go install -v ./...

.PHONY: test
test:
	go test -v ./...

.PHONY: vet
vet:
	go vet -all ./...
