#!/bin/bash
set -e

source venv/bin/activate

echo "Running benchmarks..."
python python/scripts/benchmark_all.py
