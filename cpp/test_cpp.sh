#!/bin/bash

echo "============================================================"
echo "C++ Inference Tests"
echo "============================================================"

cd "$(dirname "$0")"

# Test 1: Compilation
echo "Test 1: Compilation"
if [ -f "simple_inference" ]; then
    echo "✓ Executable exists"
    PASS=1
else
    echo "✗ Executable missing"
    PASS=0
fi

# Test 2: Run with test image
echo ""
echo "Test 2: Preprocessing pipeline"
if [ -f "test_image.jpg" ]; then
    OUTPUT=$(./simple_inference test_image.jpg 2>&1)
    if echo "$OUTPUT" | grep -q "Preprocessing test PASSED"; then
        echo "✓ Preprocessing test passed"
        ((PASS++))
    else
        echo "✗ Preprocessing test failed"
    fi
else
    echo "✗ Test image missing"
fi

# Test 3: Error handling
echo ""
echo "Test 3: Error handling"
OUTPUT=$(./simple_inference nonexistent.jpg 2>&1)
if echo "$OUTPUT" | grep -q "Error"; then
    echo "✓ Error handling works"
    ((PASS++))
else
    echo "✗ Error handling failed"
fi

echo ""
echo "============================================================"
echo "C++ Tests: $PASS/3 passed"
echo "============================================================"

[ $PASS -eq 3 ] && exit 0 || exit 1
