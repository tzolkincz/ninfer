# NInfer build with local patches.
#
#   make / make build   apply local patches, configure and build the release preset
#   make dev            same, but with tests and benchmarks (dev preset)
#   make test           build the dev preset and run ctest
#   make patch          apply patches/*.patch (idempotent, no-op if already applied)
#   make unpatch        reverse-apply the local patches
#   make status         show the patch state of the tree
#   make clean          remove the build directory
#
# Both presets use the shared build/ directory; switching between `build` and `dev`
# reconfigures it. Patches are applied with `git apply` and detected as
# already-applied via a reverse check, so the targets are safe to re-run and to
# run after `git pull` / `git checkout -- .`.

JOBS  ?= $(shell nproc)
CMAKE ?= cmake
NINJA ?= ninja
PATCHES := $(sort $(wildcard patches/*.patch))

.PHONY: all build dev test patch unpatch status clean help
.DEFAULT_GOAL := build

all: build

build: patch
	$(CMAKE) --preset release
	$(NINJA) -C build -j$(JOBS)

dev: patch
	$(CMAKE) --preset dev
	$(NINJA) -C build -j$(JOBS)

test: dev
	ctest --test-dir build --output-on-failure

patch:
	@set -e; \
	for p in $(PATCHES); do \
		if git apply --check "$$p" >/dev/null 2>&1; then \
			echo "applying $$p"; \
			git apply "$$p"; \
		elif git apply -R --check "$$p" >/dev/null 2>&1; then \
			echo "$$p: already applied"; \
		else \
			echo "error: $$p applies neither forward nor reverse" >&2; \
			echo "hint: the tree moved under the patch; run 'make unpatch' (or regenerate the patch)" >&2; \
			exit 1; \
		fi; \
	done

unpatch:
	@set -e; \
	for p in $(PATCHES); do \
		if git apply -R --check "$$p" >/dev/null 2>&1; then \
			echo "reverting $$p"; \
			git apply -R "$$p"; \
		else \
			echo "error: $$p is not applied; nothing to revert" >&2; \
			exit 1; \
		fi; \
	done

status:
	@for p in $(PATCHES); do \
		if git apply -R --check "$$p" >/dev/null 2>&1; then \
			echo "applied  $$p"; \
		elif git apply --check "$$p" >/dev/null 2>&1; then \
			echo "pending  $$p"; \
		else \
			echo "conflict $$p"; \
		fi; \
	done

clean:
	rm -rf build

help:
	@echo "targets: build (default), dev, test, patch, unpatch, status, clean"
