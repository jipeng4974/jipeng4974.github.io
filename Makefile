HUGO ?= hugo
NPM ?= npm
BIND ?= 127.0.0.1
PORT ?=
THEME_DIR ?= ../oink

.PHONY: default b build d dev p publish s serve workspace

default: dev

b: build
build: workspace
	@HUGO_MODULE_WORKSPACE="$(CURDIR)/go.work" $(NPM) run build

p: publish
publish: workspace
	@HUGO_MODULE_WORKSPACE="$(CURDIR)/go.work" $(HUGO) \
		--cleanDestinationDir \
		--logLevel info \
		--gc \
		--minify \
		-d docs
	@touch docs/.nojekyll
	@echo "Site published to docs/ — commit and push, then Pages serves it from main:/docs"

d: dev

dev: workspace
	@HUGO_MODULE_WORKSPACE="$(CURDIR)/go.work" $(HUGO) server \
		--cleanDestinationDir \
		--logLevel info \
		--environment dev \
		-DFE \
		--printPathWarnings \
		--disableFastRender \
		--renderToMemory \
		--minify \
		--bind "$(BIND)" $(if $(strip $(PORT)),--port "$(PORT)")

s: serve
serve: workspace
	@HUGO_MODULE_WORKSPACE="$(CURDIR)/go.work" $(NPM) run serve -- --bind "$(BIND)" $(if $(strip $(PORT)),--port "$(PORT)")

workspace:
	@test -f "$(THEME_DIR)/go.mod" || { \
		echo "Theme not found: $(THEME_DIR)" >&2; \
		exit 1; \
	}
	@test -f go.work || go work init .
	@go work use .
	@go work edit -replace=github.com/pgsty/oink="$(THEME_DIR)"
