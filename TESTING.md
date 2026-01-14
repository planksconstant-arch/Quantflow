## 🧪 Testing & Quality

QuantFlow includes a comprehensive test suite to ensure reliability:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=models --cov=analysis --cov-report=html
```

**Test Coverage:**
- ✅ Black-Scholes pricing validation
- ✅ Greeks calculation accuracy
- ✅ Input validation & edge cases
- ✅ Monte Carlo convergence tests
- ✅ ML model robustness

**Continuous Integration:**
- Automated testing on every commit via GitHub Actions
- Python 3.10+ compatibility tested
- Code quality checks with flake8
