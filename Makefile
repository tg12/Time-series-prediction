.PHONY: format lint style test docs help

# Directories to run quality checks on
CHECK_DIRS := tfts examples tests

## Format Python files with Black
format:  ## Format Python sources
	black $(CHECK_DIRS)

## Run static analysis checks
lint:  ## Run Ruff, Black, and pre-commit checks
	ruff check .
	black --check $(CHECK_DIRS)
	pre-commit run --all-files

## Run formatters and linters
style: format lint  ## Format code and run linting tools

## Run all unit tests
test:  ## Run the test suite with pytest
	python3 -m pytest -q

## Build the documentation
docs:  ## Build HTML documentation using Sphinx
	make -C docs clean M=$(shell pwd)
	make -C docs html M=$(shell pwd)

## Display help for make targets
help:  ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[33m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z\/_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
