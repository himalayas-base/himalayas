# HiMaLAYAS Makefile

.PHONY: help format lint test build publish clean

format: ## Format Python files using Black
	black --line-length=100 src/
lint: ## Run flake8 and pylint (best-effort, non-fatal)
	flake8 src/ || true
	pylint src/ || true
test: ## Run test suite (if present)
	pip install -e . > /dev/null
	pytest -vv --tb=auto || true
build: ## Build source and wheel distributions
	python -m build
publish: ## Publish package to PyPI (requires build)
	twine upload dist/*
	make clean
clean: ## Remove build artifacts and caches
	find . \( \
		-name ".DS_Store" -o \
		-name ".ipynb_checkpoints" -o \
		-name "__pycache__" -o \
		-name ".pytest_cache" \
	\) -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info src/*.egg-info
help: ## Show available make targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*##"}; {printf "%-12s %s\n", $$1, $$2}'
