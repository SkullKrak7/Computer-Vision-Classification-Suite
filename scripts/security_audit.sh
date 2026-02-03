#!/bin/bash
# Security audit script

set -e

echo "Running security audit..."

# 1. Bandit - Python security scanner
echo "1. Running Bandit..."
bandit -r backend/app python/src -f json -o security-report.json || true
bandit -r backend/app python/src

# 2. Safety - Check dependencies for vulnerabilities
echo ""
echo "2. Checking dependencies with pip-audit..."
pip-audit --format json --output vulnerability-report.json || true
pip-audit

# 3. Check for secrets
echo ""
echo "3. Scanning for secrets..."
detect-secrets scan --baseline .secrets.baseline

echo ""
echo "Security audit complete!"
echo "Reports: security-report.json, vulnerability-report.json"
