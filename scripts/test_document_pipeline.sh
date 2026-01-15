#!/bin/bash
#==============================================================================
# Document Pipeline Test Script
#==============================================================================
# This script tests the document processing pipeline:
#   1. Upload a document
#   2. Process/chunk the document
#   3. Generate embeddings
#   4. Store in vector database
#   5. Search for content
#   6. Delete the document
#
# Requirements:
#   - Python service running on localhost:8001
#   - llama.cpp embedding model running on localhost:8083
#   - MongoDB accessible
#   - jq installed for JSON parsing
#
# Usage: ./test_document_pipeline.sh [document_path]
#
# Default test uses documents from testing_data/Reference folder
#==============================================================================

set -e

# Configuration
PYTHON_SERVER="http://localhost:8001"
EMBEDDING_SERVER="http://localhost:8083"
TESTING_DATA_DIR="/data/projects/llm_website/testing_data"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Timing
declare -A STEP_TIMES
TOTAL_START=0

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

start_timer() { echo $(date +%s%3N); }
end_timer() {
    local start=$1
    local end=$(date +%s%3N)
    echo $((end - start))
}
format_duration() {
    local ms=$1
    if [ $ms -lt 1000 ]; then
        echo "${ms}ms"
    else
        echo "$((ms/1000)).$((ms%1000))s"
    fi
}

check_services() {
    log_step "CHECKING SERVICES"

    # Check embedding server
    log_info "Checking embedding server at ${EMBEDDING_SERVER}..."
    if curl -s "${EMBEDDING_SERVER}/health" > /dev/null 2>&1; then
        log_success "Embedding server is running"
    else
        log_warning "Embedding server not responding (will use Python fallback)"
    fi

    # Check Python service
    log_info "Checking Python service at ${PYTHON_SERVER}..."
    if curl -s "${PYTHON_SERVER}/health" > /dev/null 2>&1; then
        log_success "Python service is running"
    else
        log_error "Python service not responding"
        exit 1
    fi
}

test_embedding_direct() {
    log_step "TESTING EMBEDDING SERVICE DIRECTLY"

    local test_text="This is a test sentence for embedding generation."
    local start=$(start_timer)

    log_info "Testing embedding generation..."

    local response=$(curl -s -X POST "${EMBEDDING_SERVER}/embedding" \
        -H "Content-Type: application/json" \
        -d "{\"content\": \"$test_text\"}" 2>&1)

    local elapsed=$(end_timer $start)
    STEP_TIMES["embedding_direct"]=$elapsed

    if echo "$response" | jq -e '.embedding' > /dev/null 2>&1; then
        local dim=$(echo "$response" | jq '.embedding | length')
        log_success "Embedding generated ($(format_duration $elapsed))"
        log_info "Dimensions: $dim"
    else
        log_warning "Direct embedding failed, will use Python service"
        log_info "Response: $response"
    fi
}

test_document_upload() {
    local doc_path="$1"
    log_step "1. UPLOAD DOCUMENT"

    local start=$(start_timer)

    log_info "Uploading: $doc_path"
    log_info "File size: $(du -h "$doc_path" | cut -f1)"

    local response=$(curl -s -X POST "${PYTHON_SERVER}/documents/upload" \
        -F "file=@${doc_path}" \
        -H "Accept: application/json" 2>&1)

    local elapsed=$(end_timer $start)
    STEP_TIMES["upload"]=$elapsed

    if echo "$response" | jq -e '.success == true or .document_id' > /dev/null 2>&1; then
        DOCUMENT_ID=$(echo "$response" | jq -r '.document_id // .id')
        log_success "Document uploaded ($(format_duration $elapsed))"
        log_info "Document ID: $DOCUMENT_ID"
    else
        log_error "Upload failed: $response"
        exit 1
    fi
}

test_document_process() {
    log_step "2. PROCESS DOCUMENT (CHUNKING + EMBEDDING)"

    local start=$(start_timer)

    log_info "Processing document ID: $DOCUMENT_ID"
    log_info "This includes chunking and embedding generation..."

    local response=$(curl -s -X POST "${PYTHON_SERVER}/documents/process" \
        -H "Content-Type: application/json" \
        -d "{\"document_id\": \"$DOCUMENT_ID\"}" 2>&1)

    local elapsed=$(end_timer $start)
    STEP_TIMES["process"]=$elapsed

    if echo "$response" | jq -e '.success == true or .chunks' > /dev/null 2>&1; then
        local chunks=$(echo "$response" | jq -r '.chunks_created // .chunks // "N/A"')
        log_success "Document processed ($(format_duration $elapsed))"
        log_info "Chunks created: $chunks"
    else
        log_warning "Process endpoint may not exist, trying alternative..."
        # Document might be processed on upload
    fi
}

test_document_search() {
    log_step "3. SEARCH FOR DOCUMENT CONTENT"

    local start=$(start_timer)

    # Search for something that should be in the document
    local search_query="cotton bale processing"

    log_info "Searching for: \"$search_query\""

    local response=$(curl -s -X POST "${PYTHON_SERVER}/documents/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$search_query\", \"limit\": 5}" 2>&1)

    local elapsed=$(end_timer $start)
    STEP_TIMES["search"]=$elapsed

    if echo "$response" | jq -e '.results or .documents' > /dev/null 2>&1; then
        local count=$(echo "$response" | jq '.results | length // .documents | length // 0')
        log_success "Search completed ($(format_duration $elapsed))"
        log_info "Results found: $count"

        if [ "$count" -gt 0 ]; then
            echo -e "${YELLOW}Top result:${NC}"
            echo "$response" | jq -r '.results[0].content // .documents[0].content // "N/A"' | head -5
        fi
    else
        log_warning "Search may have returned empty or different format"
        log_info "Response: $(echo "$response" | head -c 200)"
    fi
}

test_vector_search() {
    log_step "4. VECTOR SIMILARITY SEARCH"

    local start=$(start_timer)

    local search_query="How does cotton bale loan processing work?"

    log_info "Vector search for: \"$search_query\""

    local response=$(curl -s -X POST "${PYTHON_SERVER}/query/search" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$search_query\", \"search_type\": \"documents\", \"limit\": 5}" 2>&1)

    local elapsed=$(end_timer $start)
    STEP_TIMES["vector_search"]=$elapsed

    if echo "$response" | jq -e '.results' > /dev/null 2>&1; then
        local count=$(echo "$response" | jq '.results | length')
        log_success "Vector search completed ($(format_duration $elapsed))"
        log_info "Results: $count"

        if [ "$count" -gt 0 ]; then
            local top_score=$(echo "$response" | jq -r '.results[0].similarity // .results[0].score // "N/A"')
            log_info "Top similarity score: $top_score"
        fi
    else
        log_info "Vector search response: $(echo "$response" | head -c 200)"
    fi
}

test_document_delete() {
    log_step "5. DELETE DOCUMENT"

    if [ -z "$DOCUMENT_ID" ]; then
        log_warning "No document ID to delete"
        return
    fi

    local start=$(start_timer)

    log_info "Deleting document ID: $DOCUMENT_ID"

    local response=$(curl -s -X DELETE "${PYTHON_SERVER}/documents/${DOCUMENT_ID}" 2>&1)

    local elapsed=$(end_timer $start)
    STEP_TIMES["delete"]=$elapsed

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        log_success "Document deleted ($(format_duration $elapsed))"
    else
        log_warning "Delete response: $response"
    fi
}

print_metrics() {
    log_step "PERFORMANCE METRICS"

    local total_end=$(date +%s%3N)
    local total_time=$((total_end - TOTAL_START))

    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║            DOCUMENT PIPELINE TEST METRICS                   ║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"

    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Embedding Direct:" "$(format_duration ${STEP_TIMES["embedding_direct"]:-0})"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Upload:" "$(format_duration ${STEP_TIMES["upload"]:-0})"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Process:" "$(format_duration ${STEP_TIMES["process"]:-0})"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Search:" "$(format_duration ${STEP_TIMES["search"]:-0})"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Vector Search:" "$(format_duration ${STEP_TIMES["vector_search"]:-0})"
    printf "${CYAN}║${NC}   %-25s %30s ${CYAN}║${NC}\n" "Delete:" "$(format_duration ${STEP_TIMES["delete"]:-0})"

    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    printf "${CYAN}║${NC}   ${GREEN}%-25s %30s${NC} ${CYAN}║${NC}\n" "TOTAL TIME" "$(format_duration $total_time)"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
}

main() {
    TOTAL_START=$(date +%s%3N)

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         DOCUMENT PIPELINE TEST SCRIPT                       ║${NC}"
    echo -e "${CYAN}║                                                             ║${NC}"
    echo -e "${CYAN}║  Tests the document processing pipeline including:          ║${NC}"
    echo -e "${CYAN}║  - Embedding generation                                     ║${NC}"
    echo -e "${CYAN}║  - Document upload and chunking                             ║${NC}"
    echo -e "${CYAN}║  - Vector similarity search                                 ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Get document path
    local doc_path="$1"

    if [ -z "$doc_path" ]; then
        # Use default test document
        doc_path="${TESTING_DATA_DIR}/Reference/ProviderSystem/Cotton Bale Loan Processing.docx"
        if [ ! -f "$doc_path" ]; then
            # Try alternative
            doc_path=$(find "${TESTING_DATA_DIR}" -name "*.docx" -o -name "*.pdf" | head -1)
        fi
    fi

    if [ ! -f "$doc_path" ]; then
        log_error "No test document found at: $doc_path"
        log_info "Usage: $0 [document_path]"
        log_info "Or ensure testing_data folder has documents"
        exit 1
    fi

    log_info "Using test document: $doc_path"

    check_services
    test_embedding_direct
    test_document_upload "$doc_path"
    test_document_process
    test_document_search
    test_vector_search
    test_document_delete
    print_metrics

    echo ""
    log_success "Document Pipeline tests completed!"
    echo ""
}

main "$@"
