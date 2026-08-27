# Contributing to QuantFlow

Thank you for your interest in contributing to QuantFlow. This document outlines the guidelines and workflow for contributing to the repository.

---

## Development Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git
- Docker (optional, for containerized workflows)

### 2. Environment Installation
```bash
# Clone the repository
git clone https://github.com/planksconstant-arch/Quantflow.git
cd Quantflow

# Create and activate a virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Code Quality and Style Guidelines

1. **PEP 8 Compliance**:
   - Run `flake8` before submitting any pull request:
   ```bash
   flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude=.venv,__pycache__
   ```

2. **Zero-Emoji Policy**:
   - Do not use emojis in docstrings, user interfaces, documentation, or commit messages. Maintain clean institutional quantitative formatting.

3. **Type Annotations**:
   - Annotate function parameters and return types across all public interfaces in `models/`, `data/`, `analysis/`, and `utils/`.

---

## Testing & Benchmarks

### Running the Test Suite
Ensure all unit tests pass before opening a PR:
```bash
pytest tests/ -v --cov=models --cov=analysis --cov=utils --cov=data
```

### Running the Latency Benchmark
Verify microsecond performance benchmarks:
```bash
python tests/benchmark_latency.py
```

---

## Pull Request Process

1. **Branch Naming**:
   - `feature/<feature-name>` for new capabilities
   - `fix/<bug-description>` for bug fixes
   - `perf/<optimization>` for latency or memory improvements
   - `docs/<doc-update>` for documentation updates

2. **Commit Guidelines**:
   - Write clear, imperative commit messages (e.g., `Add Almgren-Chriss nonlinear execution trajectory`).
   - Ensure commits do not include emojis.

3. **Submitting PRs**:
   - Open a pull request against the `main` branch.
   - Fill out the PR template completely.
   - Verify that all CI checks (linting, test coverage, and latency benchmarks) pass.

---

## License
By contributing to QuantFlow, you agree that your contributions will be licensed under the project's MIT License.
