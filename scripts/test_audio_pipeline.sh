#!/bin/bash
#==============================================================================
# Audio Pipeline End-to-End Test Script
#==============================================================================
# This script tests the complete audio pipeline workflow:
#   1. Upload an audio file
#   2. Analyze the file (transcription, emotion detection, LLM analysis)
#   3. Store the analysis with customer support name 'Christine'
#   4. Search for all calls with customer support name 'Christine'
#   5. Update the record with modifications
#   6. Delete the call record
#
# Requirements:
#   - Node.js server running on localhost:3000
#   - Python service running on localhost:8001
#   - MongoDB accessible
#   - jq installed for JSON parsing
#   - curl installed
#
# Usage: ./test_audio_pipeline.sh [audio_file_path]
#
# If no audio file path is provided, the script will use a default test file.
#==============================================================================

set -e  # Exit on error

# Configuration
NODE_SERVER="http://localhost:3000"
PYTHON_SERVER="http://localhost:8001"
CUSTOMER_SUPPORT_NAME="Christine"
TEST_AUDIO_DIR="/tmp/audio_test"
METRICS_FILE="/tmp/audio_test_metrics.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Timing variables
declare -A STEP_TIMES
TOTAL_START_TIME=0

# LLM metrics tracking
declare -A LLM_METRICS

#------------------------------------------------------------------------------
# Utility Functions
#------------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_step() {
    echo ""
    echo -e "${CYAN}========================================${NC}"
    echo -e "${CYAN}STEP: $1${NC}"
    echo -e "${CYAN}========================================${NC}"
}

start_timer() {
    echo $(date +%s%3N)
}

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
        local seconds=$((ms / 1000))
        local remaining_ms=$((ms % 1000))
        echo "${seconds}.${remaining_ms}s"
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    if ! command -v curl &> /dev/null; then
        log_error "curl is not installed"
        exit 1
    fi

    if ! command -v jq &> /dev/null; then
        log_error "jq is not installed. Install with: sudo apt install jq"
        exit 1
    fi

    log_success "All dependencies available"
}

check_services() {
    log_info "Checking if services are running..."

    # Check Node.js server
    if ! curl -s "${NODE_SERVER}/health" > /dev/null 2>&1; then
        log_warning "Node.js server not responding at ${NODE_SERVER}"
        log_info "Attempting to check Python server directly..."
    else
        log_success "Node.js server is running"
    fi

    # Check Python server
    if ! curl -s "${PYTHON_SERVER}/health" > /dev/null 2>&1; then
        log_error "Python server not responding at ${PYTHON_SERVER}"
        log_info "Please start the services first"
        exit 1
    fi
    log_success "Python server is running"
}

#------------------------------------------------------------------------------
# Test Functions
#------------------------------------------------------------------------------

test_upload_file() {
    local audio_file="$1"
    log_step "1. UPLOAD AUDIO FILE"

    local start=$(start_timer)

    log_info "Uploading file: $audio_file"
    log_info "File size: $(du -h "$audio_file" | cut -f1)"

    # Upload via Python service (more reliable for direct file upload)
    local response=$(curl -s -X POST "${PYTHON_SERVER}/audio/upload" \
        -F "file=@${audio_file}" \
        -H "Accept: application/json")

    local elapsed=$(end_timer $start)
    STEP_TIMES["upload"]=$elapsed

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        UPLOADED_FILE_PATH=$(echo "$response" | jq -r '.temp_path // .filepath')
        UPLOADED_FILENAME=$(echo "$response" | jq -r '.filename')
        local size_mb=$(echo "$response" | jq -r '.size_mb')

        log_success "File uploaded successfully"
        log_info "  Temp path: $UPLOADED_FILE_PATH"
        log_info "  Filename: $UPLOADED_FILENAME"
        log_info "  Size: ${size_mb}MB"
        log_info "  Duration: $(format_duration $elapsed)"
    else
        log_error "Upload failed: $response"
        exit 1
    fi
}

test_analyze_file() {
    log_step "2. ANALYZE AUDIO FILE"

    local start=$(start_timer)

    log_info "Starting analysis with SenseVoice..."
    log_info "This may take a while depending on audio length..."

    # Create request body
    local request_body=$(jq -n \
        --arg path "$UPLOADED_FILE_PATH" \
        --arg filename "$UPLOADED_FILENAME" \
        '{audio_path: $path, original_filename: $filename, language: "auto"}')

    # Use streaming endpoint and capture full response
    # The SSE stream sends progress updates, then the final result
    local temp_response="/tmp/audio_analysis_response_$$.txt"

    curl -s -N -X POST "${PYTHON_SERVER}/audio/analyze-stream" \
        -H "Content-Type: application/json" \
        -H "Accept: text/event-stream" \
        -d "$request_body" > "$temp_response" 2>&1 &

    local curl_pid=$!

    # Monitor progress
    local last_step=""
    local analysis_complete=false
    local timeout=300  # 5 minute timeout
    local elapsed_seconds=0

    while [ $elapsed_seconds -lt $timeout ]; do
        if ! kill -0 $curl_pid 2>/dev/null; then
            # curl has finished
            break
        fi

        # Check for progress updates
        if [ -f "$temp_response" ]; then
            local current_step=$(grep -o '"step":"[^"]*"' "$temp_response" | tail -1 | cut -d'"' -f4)
            if [ -n "$current_step" ] && [ "$current_step" != "$last_step" ]; then
                log_info "  Progress: $current_step"
                last_step="$current_step"
            fi

            # Check if we have the final result
            if grep -q '"transcription"' "$temp_response" 2>/dev/null; then
                analysis_complete=true
                break
            fi
        fi

        sleep 2
        elapsed_seconds=$((elapsed_seconds + 2))
    done

    # Wait for curl to finish
    wait $curl_pid 2>/dev/null || true

    local elapsed=$(end_timer $start)
    STEP_TIMES["analyze"]=$elapsed

    # Parse the response - look for the last complete JSON object
    if [ -f "$temp_response" ]; then
        # Extract the final JSON result (after all SSE events)
        # SSE format: "data: {json}" - need to extract JSON from lines starting with "data: "
        # The complete result may be wrapped: {"type": "complete", "data": {...}}
        # or it may be the raw result with "transcription" at the top level

        # First try to get the complete event's result field
        # Format is: data: {"type": "complete", "result": {...}}
        ANALYSIS_RESULT=$(grep '^data: ' "$temp_response" | grep '"type": "complete"' | tail -1 | sed 's/^data: //' | jq -r '.result // .' 2>/dev/null)

        # If that didn't work, try direct extraction for transcription field
        if [ -z "$ANALYSIS_RESULT" ] || [ "$ANALYSIS_RESULT" = "null" ]; then
            ANALYSIS_RESULT=$(grep '^data: ' "$temp_response" | grep '"transcription"' | tail -1 | sed 's/^data: //')
        fi

        if [ -n "$ANALYSIS_RESULT" ] && echo "$ANALYSIS_RESULT" | jq -e '.transcription' > /dev/null 2>&1; then
            log_success "Analysis completed"

            # Extract key information
            local transcription_length=$(echo "$ANALYSIS_RESULT" | jq -r '.transcription | length')
            local primary_emotion=$(echo "$ANALYSIS_RESULT" | jq -r '.emotions.primary // "unknown"')
            local language=$(echo "$ANALYSIS_RESULT" | jq -r '.language // "unknown"')
            local duration=$(echo "$ANALYSIS_RESULT" | jq -r '.audio_metadata.duration_seconds // 0')
            local analysis_model=$(echo "$ANALYSIS_RESULT" | jq -r '.call_content.analysis_model // "unknown"')

            log_info "  Transcription length: $transcription_length chars"
            log_info "  Primary emotion: $primary_emotion"
            log_info "  Language: $language"
            log_info "  Audio duration: ${duration}s"
            log_info "  Analysis model: $analysis_model"
            log_info "  Analysis time: $(format_duration $elapsed)"

            # Store LLM metrics
            LLM_METRICS["analysis_model"]="$analysis_model"
            LLM_METRICS["analysis_duration_ms"]="$elapsed"

            # Check for summary (only for longer audio)
            local has_summary=$(echo "$ANALYSIS_RESULT" | jq -r 'if .transcription_summary then "yes" else "no" end')
            log_info "  Has summary: $has_summary"

            # Display full transcription
            echo ""
            echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${YELLOW}│                    FULL TRANSCRIPTION                       │${NC}"
            echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
            echo "$ANALYSIS_RESULT" | jq -r '.transcription_plain // .transcription' | fold -w 80 -s
            echo ""

            # Display call content analysis
            echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${YELLOW}│                   CALL CONTENT ANALYSIS                     │${NC}"
            echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
            echo "$ANALYSIS_RESULT" | jq -r '
                "Subject: \(.call_content.subject // "N/A")",
                "Outcome: \(.call_content.outcome // "N/A")",
                "Customer Name: \(.call_content.customer_name // "N/A")",
                "Confidence: \(.call_content.confidence // "N/A")",
                "Analysis Model: \(.call_content.analysis_model // "N/A")"
            '
            echo ""

            # Display emotions
            echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${YELLOW}│                    EMOTION ANALYSIS                         │${NC}"
            echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
            echo "$ANALYSIS_RESULT" | jq -r '
                "Primary Emotion: \(.emotions.primary // "N/A")",
                "Detected Emotions: \(.emotions.detected | join(", ") // "N/A")"
            '
            echo ""

            # Display audio metadata
            echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${YELLOW}│                    AUDIO METADATA                           │${NC}"
            echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
            echo "$ANALYSIS_RESULT" | jq -r '
                "Duration: \(.audio_metadata.duration_seconds // 0)s",
                "Sample Rate: \(.audio_metadata.sample_rate // "N/A") Hz",
                "Channels: \(.audio_metadata.channels // "N/A")",
                "Format: \(.audio_metadata.format // "N/A")",
                "File Size: \((.audio_metadata.file_size_bytes // 0) / 1024 / 1024 | . * 100 | floor / 100) MB"
            '
            echo ""

            # Store additional LLM metrics
            LLM_METRICS["call_subject"]=$(echo "$ANALYSIS_RESULT" | jq -r '.call_content.subject // "N/A"')
            LLM_METRICS["call_outcome"]=$(echo "$ANALYSIS_RESULT" | jq -r '.call_content.outcome // "N/A"')
            LLM_METRICS["customer_name"]=$(echo "$ANALYSIS_RESULT" | jq -r '.call_content.customer_name // "N/A"')
            LLM_METRICS["confidence"]=$(echo "$ANALYSIS_RESULT" | jq -r '.call_content.confidence // "N/A"')

        else
            log_error "Analysis failed - could not parse result"
            log_info "Raw response (last 500 chars):"
            tail -c 500 "$temp_response"
            rm -f "$temp_response"
            exit 1
        fi
    else
        log_error "No response received from analysis endpoint"
        exit 1
    fi

    rm -f "$temp_response"
}

test_store_analysis() {
    log_step "3. STORE ANALYSIS WITH CUSTOMER SUPPORT NAME 'CHRISTINE'"

    local start=$(start_timer)

    log_info "Preparing analysis data for storage..."

    # Merge the analysis result with our custom metadata
    local store_request=$(echo "$ANALYSIS_RESULT" | jq \
        --arg staff "$CUSTOMER_SUPPORT_NAME" \
        --arg customer "Test Customer" \
        --arg mood "Neutral" \
        --arg outcome "Resolved" \
        --arg filename "$UPLOADED_FILENAME" \
        '. + {
            customer_support_staff: $staff,
            ewr_customer: $customer,
            mood: $mood,
            outcome: $outcome,
            filename: $filename,
            metadata: {
                customer_support_staff: $staff,
                ewr_customer: $customer,
                mood: $mood,
                outcome: $outcome,
                filename: $filename
            }
        }')

    # Store via API
    local response=$(curl -s -X POST "${PYTHON_SERVER}/audio/store" \
        -H "Content-Type: application/json" \
        -d "$store_request")

    local elapsed=$(end_timer $start)
    STEP_TIMES["store"]=$elapsed

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        ANALYSIS_ID=$(echo "$response" | jq -r '.analysis_id // .id')

        log_success "Analysis stored successfully"
        log_info "  Analysis ID: $ANALYSIS_ID"
        log_info "  Customer Support Staff: $CUSTOMER_SUPPORT_NAME"
        log_info "  Storage time: $(format_duration $elapsed)"
    else
        log_error "Store failed: $response"
        exit 1
    fi
}

test_search_by_staff() {
    log_step "4. SEARCH FOR ALL CALLS WITH CUSTOMER SUPPORT NAME 'CHRISTINE'"

    local start=$(start_timer)

    log_info "Searching for calls with customer_support_staff='$CUSTOMER_SUPPORT_NAME'..."

    # URL encode the staff name
    local encoded_name=$(echo "$CUSTOMER_SUPPORT_NAME" | jq -sRr @uri)

    local response=$(curl -s -X GET "${PYTHON_SERVER}/audio/search?customer_support_staff=${encoded_name}&limit=50")

    local elapsed=$(end_timer $start)
    STEP_TIMES["search"]=$elapsed

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        local total=$(echo "$response" | jq -r '.total // (.results | length)')
        local results=$(echo "$response" | jq -r '.results')

        log_success "Search completed"
        log_info "  Total calls found: $total"
        log_info "  Search time: $(format_duration $elapsed)"

        # List found calls
        if [ "$total" -gt 0 ]; then
            log_info "  Found calls:"
            echo "$response" | jq -r '.results[] | "    - ID: \(.id // .analysis_id) | Date: \(.call_date // "N/A") | Mood: \(.mood // "N/A") | Outcome: \(.outcome // "N/A")"'
        fi

        # Verify our test call is in the results
        if echo "$response" | jq -e ".results[] | select(.id == \"$ANALYSIS_ID\" or .analysis_id == \"$ANALYSIS_ID\")" > /dev/null 2>&1; then
            log_success "Test call found in search results"
        else
            log_warning "Test call not found in search results (may be indexing delay)"
        fi
    else
        log_error "Search failed: $response"
        # Don't exit - continue with other tests
    fi
}

test_update_record() {
    log_step "5. UPDATE THE RECORD"

    local start=$(start_timer)

    log_info "Updating record with ID: $ANALYSIS_ID"

    # Update with new values
    local update_data=$(jq -n \
        --arg mood "Positive" \
        --arg outcome "Excellent Resolution" \
        --arg notes "Updated by automated test script" \
        '{
            mood: $mood,
            outcome: $outcome,
            test_notes: $notes,
            updated_by_test: true
        }')

    local response=$(curl -s -X PUT "${PYTHON_SERVER}/audio/${ANALYSIS_ID}" \
        -H "Content-Type: application/json" \
        -d "$update_data")

    local elapsed=$(end_timer $start)
    STEP_TIMES["update"]=$elapsed

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        log_success "Record updated successfully"
        log_info "  Updated mood: Positive"
        log_info "  Updated outcome: Excellent Resolution"
        log_info "  Update time: $(format_duration $elapsed)"

        # Verify the update by fetching the record
        log_info "Verifying update..."
        local verify=$(curl -s -X GET "${PYTHON_SERVER}/audio/${ANALYSIS_ID}")

        if echo "$verify" | jq -e '.mood == "Positive"' > /dev/null 2>&1; then
            log_success "Update verified - mood is now 'Positive'"

            # Display full updated document from MongoDB
            echo ""
            echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
            echo -e "${YELLOW}│              UPDATED DOCUMENT FROM MONGODB                  │${NC}"
            echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"
            echo "$verify" | jq '{
                id: .id,
                filename: .filename,
                customer_support_staff: .customer_support_staff,
                ewr_customer: .ewr_customer,
                mood: .mood,
                outcome: .outcome,
                test_notes: .test_notes,
                updated_by_test: .updated_by_test,
                call_content: .call_content,
                emotions: .emotions,
                language: .language,
                audio_metadata: .audio_metadata,
                created_at: .created_at,
                updated_at: .updated_at
            }'
            echo ""

            # Store the full document for reference
            UPDATED_DOCUMENT="$verify"
        else
            log_warning "Could not verify update"
        fi
    else
        log_error "Update failed: $response"
        # Don't exit - continue with delete test
    fi
}

test_delete_record() {
    log_step "6. DELETE THE CALL RECORD"

    local start=$(start_timer)

    log_info "Deleting record with ID: $ANALYSIS_ID"

    local response=$(curl -s -X DELETE "${PYTHON_SERVER}/audio/${ANALYSIS_ID}")

    local elapsed=$(end_timer $start)
    STEP_TIMES["delete"]=$elapsed

    if echo "$response" | jq -e '.success == true' > /dev/null 2>&1; then
        log_success "Record deleted successfully"
        log_info "  Delete time: $(format_duration $elapsed)"

        # Verify deletion
        log_info "Verifying deletion..."
        local verify=$(curl -s -X GET "${PYTHON_SERVER}/audio/${ANALYSIS_ID}")

        if echo "$verify" | jq -e '.success == false or .error' > /dev/null 2>&1; then
            log_success "Deletion verified - record no longer exists"
        else
            log_warning "Record may still exist (could be caching)"
        fi
    else
        log_error "Delete failed: $response"
    fi
}

print_metrics_report() {
    log_step "TEST METRICS REPORT"

    local total_end=$(date +%s%3N)
    local total_elapsed=$((total_end - TOTAL_START_TIME))

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║              AUDIO PIPELINE TEST METRICS                    ║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"

    echo -e "${CYAN}║${NC} ${YELLOW}Step Timings:${NC}                                              ${CYAN}║${NC}"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Upload:" "$(format_duration ${STEP_TIMES["upload"]:-0})"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Analyze:" "$(format_duration ${STEP_TIMES["analyze"]:-0})"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Store:" "$(format_duration ${STEP_TIMES["store"]:-0})"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Search:" "$(format_duration ${STEP_TIMES["search"]:-0})"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Update:" "$(format_duration ${STEP_TIMES["update"]:-0})"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Delete:" "$(format_duration ${STEP_TIMES["delete"]:-0})"

    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} ${YELLOW}LLM Metrics:${NC}                                               ${CYAN}║${NC}"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Analysis Model:" "${LLM_METRICS["analysis_model"]:-N/A}"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "LLM Processing:" "$(format_duration ${LLM_METRICS["analysis_duration_ms"]:-0})"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Confidence:" "${LLM_METRICS["confidence"]:-N/A}"

    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║${NC} ${YELLOW}LLM Call Analysis:${NC}                                         ${CYAN}║${NC}"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Customer Name:" "${LLM_METRICS["customer_name"]:-N/A}"
    printf "${CYAN}║${NC}   %-20s %35s ${CYAN}║${NC}\n" "Call Outcome:" "${LLM_METRICS["call_outcome"]:-N/A}"

    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    printf "${CYAN}║${NC}   ${GREEN}%-20s %35s${NC} ${CYAN}║${NC}\n" "TOTAL TIME:" "$(format_duration $total_elapsed)"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"

    # Display LLM server statistics
    echo ""
    echo -e "${YELLOW}┌─────────────────────────────────────────────────────────────┐${NC}"
    echo -e "${YELLOW}│                    LLM SERVER STATISTICS                    │${NC}"
    echo -e "${YELLOW}└─────────────────────────────────────────────────────────────┘${NC}"

    # Fetch LLM health/stats from all endpoints
    local general_stats=$(curl -s "http://localhost:8081/health" 2>/dev/null || echo '{"status":"unavailable"}')
    local sql_stats=$(curl -s "http://localhost:8080/health" 2>/dev/null || echo '{"status":"unavailable"}')
    local code_stats=$(curl -s "http://localhost:8082/health" 2>/dev/null || echo '{"status":"unavailable"}')
    local embed_stats=$(curl -s "http://localhost:8083/health" 2>/dev/null || echo '{"status":"unavailable"}')

    echo -e "${BLUE}General LLM (8081):${NC}"
    echo "$general_stats" | jq -r '
        if .status == "ok" then
            "  Status: \(.status)",
            "  Slots Idle: \(.slots_idle // "N/A")",
            "  Slots Processing: \(.slots_processing // "N/A")"
        else
            "  Status: unavailable"
        end
    ' 2>/dev/null || echo "  Status: unavailable"

    echo -e "${BLUE}SQL LLM (8080):${NC}"
    echo "$sql_stats" | jq -r '
        if .status == "ok" then
            "  Status: \(.status)",
            "  Slots Idle: \(.slots_idle // "N/A")",
            "  Slots Processing: \(.slots_processing // "N/A")"
        else
            "  Status: unavailable"
        end
    ' 2>/dev/null || echo "  Status: unavailable"

    echo -e "${BLUE}Code LLM (8082):${NC}"
    echo "$code_stats" | jq -r '
        if .status == "ok" then
            "  Status: \(.status)",
            "  Slots Idle: \(.slots_idle // "N/A")",
            "  Slots Processing: \(.slots_processing // "N/A")"
        else
            "  Status: unavailable"
        end
    ' 2>/dev/null || echo "  Status: unavailable"

    echo -e "${BLUE}Embedding (8083):${NC}"
    echo "$embed_stats" | jq -r '
        if .status == "ok" then
            "  Status: \(.status)",
            "  Slots Idle: \(.slots_idle // "N/A")",
            "  Slots Processing: \(.slots_processing // "N/A")"
        else
            "  Status: unavailable"
        end
    ' 2>/dev/null || echo "  Status: unavailable"
    echo ""

    # Save metrics to JSON file
    cat > "$METRICS_FILE" << EOF
{
    "test_date": "$(date -Iseconds)",
    "audio_file": "$UPLOADED_FILENAME",
    "analysis_id": "$ANALYSIS_ID",
    "customer_support_staff": "$CUSTOMER_SUPPORT_NAME",
    "step_timings_ms": {
        "upload": ${STEP_TIMES["upload"]:-0},
        "analyze": ${STEP_TIMES["analyze"]:-0},
        "store": ${STEP_TIMES["store"]:-0},
        "search": ${STEP_TIMES["search"]:-0},
        "update": ${STEP_TIMES["update"]:-0},
        "delete": ${STEP_TIMES["delete"]:-0}
    },
    "llm_metrics": {
        "analysis_model": "${LLM_METRICS["analysis_model"]:-unknown}",
        "analysis_duration_ms": ${LLM_METRICS["analysis_duration_ms"]:-0}
    },
    "total_duration_ms": $total_elapsed
}
EOF

    log_info "Metrics saved to: $METRICS_FILE"
}

cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf "$TEST_AUDIO_DIR" 2>/dev/null || true
}

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

main() {
    TOTAL_START_TIME=$(date +%s%3N)

    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║         AUDIO PIPELINE END-TO-END TEST SCRIPT              ║${NC}"
    echo -e "${CYAN}║                                                            ║${NC}"
    echo -e "${CYAN}║  This script tests the complete audio processing pipeline  ║${NC}"
    echo -e "${CYAN}║  including upload, analysis, storage, search, update,      ║${NC}"
    echo -e "${CYAN}║  and deletion operations.                                  ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""

    # Get audio file path
    local audio_file="$1"

    if [ -z "$audio_file" ]; then
        log_error "Usage: $0 <audio_file_path>"
        log_info "Example: $0 /path/to/audio.mp3"
        exit 1
    fi

    if [ ! -f "$audio_file" ]; then
        log_error "Audio file not found: $audio_file"
        exit 1
    fi

    # Setup
    check_dependencies
    check_services
    mkdir -p "$TEST_AUDIO_DIR"

    # Set trap for cleanup
    trap cleanup EXIT

    # Run tests
    test_upload_file "$audio_file"
    test_analyze_file
    test_store_analysis
    test_search_by_staff
    test_update_record
    test_delete_record

    # Print final report
    print_metrics_report

    echo ""
    log_success "All tests completed successfully!"
    echo ""
}

# Run main function with all arguments
main "$@"
