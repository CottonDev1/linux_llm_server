#!/bin/bash
# Test script for Python service management API

echo "=========================================="
echo "Testing Python Service Management API"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PYTHON_SERVICE_URL="http://localhost:8001"
NODE_SERVICE_URL="http://localhost:3000"

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
    fi
}

# Test 1: Check if Python service is running
echo "Test 1: Check Python service availability"
if curl -s -f "${PYTHON_SERVICE_URL}/status" > /dev/null 2>&1; then
    print_result 0 "Python service is running"
else
    print_result 1 "Python service is not running"
    echo -e "${RED}ERROR: Python service must be running for these tests${NC}"
    exit 1
fi
echo ""

# Test 2: Get service status via Python API
echo "Test 2: Get service status (Python API)"
STATUS_RESPONSE=$(curl -s -X POST "${PYTHON_SERVICE_URL}/admin/service/status")
if echo "$STATUS_RESPONSE" | grep -q '"success": true'; then
    print_result 0 "Status endpoint returned success"

    # Extract and display key metrics
    UPTIME=$(echo "$STATUS_RESPONSE" | grep -o '"uptime_formatted": "[^"]*"' | cut -d'"' -f4)
    MEMORY=$(echo "$STATUS_RESPONSE" | grep -o '"memory_mb": [0-9.]*' | grep -o '[0-9.]*')
    CPU=$(echo "$STATUS_RESPONSE" | grep -o '"cpu_percent": [0-9.]*' | grep -o '[0-9.]*')
    PID=$(echo "$STATUS_RESPONSE" | grep -o '"pid": [0-9]*' | grep -o '[0-9]*')

    echo -e "${YELLOW}Metrics:${NC} Uptime=$UPTIME, Memory=${MEMORY}MB, CPU=${CPU}%, PID=$PID"
else
    print_result 1 "Status endpoint failed"
    echo "Response: $STATUS_RESPONSE"
fi
echo ""

# Test 3: Get service info
echo "Test 3: Get service info (Python API)"
INFO_RESPONSE=$(curl -s "${PYTHON_SERVICE_URL}/admin/service/info")
if echo "$INFO_RESPONSE" | grep -q '"success": true'; then
    print_result 0 "Info endpoint returned success"

    # Extract service version and platform
    VERSION=$(echo "$INFO_RESPONSE" | grep -o '"service_version": "[^"]*"' | cut -d'"' -f4)
    PLATFORM=$(echo "$INFO_RESPONSE" | grep -o '"platform": "[^"]*"' | cut -d'"' -f4)

    echo -e "${YELLOW}Info:${NC} Version=$VERSION, Platform=$PLATFORM"
else
    print_result 1 "Info endpoint failed"
    echo "Response: $INFO_RESPONSE"
fi
echo ""

# Test 4: Check Node.js integration (requires auth - may fail)
echo "Test 4: Check Node.js service status endpoint"
if curl -s -f "${NODE_SERVICE_URL}/api/admin/service/python/status" > /dev/null 2>&1; then
    NODE_STATUS=$(curl -s "${NODE_SERVICE_URL}/api/admin/service/python/status")
    if echo "$NODE_STATUS" | grep -q '"success": true'; then
        print_result 0 "Node.js status proxy works"
    else
        print_result 1 "Node.js returned error"
        echo "Response: $NODE_STATUS"
    fi
else
    echo -e "${YELLOW}⚠ SKIP${NC}: Node.js endpoint requires authentication"
fi
echo ""

# Test 5: Verify restart endpoint exists (don't actually restart)
echo "Test 5: Verify restart endpoint is available"
echo -e "${YELLOW}Note:${NC} This test only checks if the endpoint exists, it does NOT restart the service"

# OPTIONS request to check if endpoint exists
if curl -s -X OPTIONS "${PYTHON_SERVICE_URL}/admin/service/restart" -o /dev/null -w "%{http_code}" | grep -qE "200|405"; then
    print_result 0 "Restart endpoint is registered"
else
    print_result 1 "Restart endpoint not found"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo "All basic tests completed."
echo ""
echo "To manually test restart (will restart the service):"
echo "  curl -X POST ${PYTHON_SERVICE_URL}/admin/service/restart"
echo ""
echo "To test via Node.js (requires auth token):"
echo "  curl -X POST -H 'Authorization: Bearer YOUR_TOKEN' \\"
echo "    ${NODE_SERVICE_URL}/api/admin/service/python/restart"
