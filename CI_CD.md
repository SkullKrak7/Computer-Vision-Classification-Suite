# CI/CD Pipeline

## Overview
This project uses GitHub Actions for continuous integration and deployment with the following tools:
- **Black**: Code formatting
- **Ruff**: Fast Python linting
- **Pytest**: Testing framework with coverage
- **Pydantic**: Data validation
- **Pre-commit**: Git hooks for code quality

## Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)
Runs on every push and pull request to `main` and `develop` branches.

**Jobs:**
- **lint-and-format**: Checks code formatting with Black and linting with Ruff
- **test**: Runs pytest with coverage reporting

### 2. Docker Build (`.github/workflows/docker.yml`)
Builds and pushes Docker images to GitHub Container Registry on:
- Push to `main` branch
- Version tags (`v*`)

**Services built:**
- Backend (FastAPI)
- Frontend (React)
- Python (Training/Inference)

## Local Development

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
make install-hooks
```

### Commands
```bash
# Format code
make format

# Lint code
make lint

# Fix linting issues
make lint-fix

# Run tests
make test

# Run all CI checks
make ci

# Clean artifacts
make clean
```

### Pre-commit Hooks
Automatically run before each commit:
- Black formatting
- Ruff linting with auto-fix
- Trailing whitespace removal
- YAML validation
- Large file checks

To skip hooks (not recommended):
```bash
git commit --no-verify
```

## Configuration

### Black (`pyproject.toml`)
- Line length: 100
- Target: Python 3.10
- Excludes: venv, build, datasets, models

### Ruff (`pyproject.toml`)
- Line length: 100
- Selected rules: E, F, I, N, W, UP, B, C4
- Ignored: E501 (line too long), N803/N806 (naming)

### Pytest (`pyproject.toml`)
- Test paths: `python/tests/`, `backend/tests/`
- Coverage: Enabled with XML and terminal reports
- Pattern: `test_*.py`

## GitHub Actions Secrets

Required secrets for deployment:
- `AWS_ROLE_ARN`: AWS IAM role for OIDC authentication
- `GITHUB_TOKEN`: Automatically provided by GitHub

## Badge Status

Add to README.md:
```markdown
![CI](https://github.com/SkullKrak7/Computer-Vision-Classification-Suite/workflows/CI/badge.svg)
![Docker](https://github.com/SkullKrak7/Computer-Vision-Classification-Suite/workflows/Docker%20Build/badge.svg)
```

## Troubleshooting

### Black formatting fails
```bash
black python/ backend/
```

### Ruff errors
```bash
ruff check --fix python/ backend/
```

### Tests fail
```bash
pytest python/tests/ backend/tests/ -v
```

### Pre-commit issues
```bash
pre-commit run --all-files
```
