#!/bin/bash
#
# MongoDB Restore Script
# ======================
# Restores MongoDB from the latest backup archive.
#
# Usage:
#   ./mongodb_restore.sh                     # Restore from default backup location
#   ./mongodb_restore.sh /path/to/backup     # Restore from specified backup file
#
# Requirements:
#   - Docker must be running
#   - MongoDB container must exist (will be recreated if needed)
#

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-/data/backups}"
DEFAULT_BACKUP="${BACKUP_DIR}/mongo-backup.archive"
CONTAINER_NAME="${MONGODB_CONTAINER:-mongodb}"
DOCKER_IMAGE="${MONGODB_IMAGE:-mongodb/mongodb-atlas-local:8}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get backup file path
BACKUP_FILE="${1:-$DEFAULT_BACKUP}"

# Validate backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
    log_error "Backup file not found: $BACKUP_FILE"
    echo ""
    echo "Available backups in ${BACKUP_DIR}:"
    ls -la "${BACKUP_DIR}"/*.archive 2>/dev/null || echo "  No .archive files found"
    exit 1
fi

log_info "Using backup file: $BACKUP_FILE"
log_info "Backup size: $(du -h "$BACKUP_FILE" | cut -f1)"

# Check if MongoDB container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log_warn "MongoDB container '${CONTAINER_NAME}' is not running"

    # Check if container exists but is stopped
    if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        log_info "Starting existing container..."
        docker start "$CONTAINER_NAME"
    else
        log_info "Creating new MongoDB container..."
        docker run -d --name "$CONTAINER_NAME" --restart unless-stopped \
            -p 27017:27017 \
            -v mongodb_data_new:/data/db \
            "$DOCKER_IMAGE"
    fi

    # Wait for MongoDB to be ready
    log_info "Waiting for MongoDB to start..."
    for i in {1..30}; do
        if docker exec "$CONTAINER_NAME" mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; then
            break
        fi
        sleep 1
    done
fi

# Verify MongoDB is accessible
if ! docker exec "$CONTAINER_NAME" mongosh --quiet --eval "db.runCommand({ping:1})" &>/dev/null; then
    log_error "Cannot connect to MongoDB in container '${CONTAINER_NAME}'"
    exit 1
fi

log_info "MongoDB is ready"

# Get document count before restore
BEFORE_COUNT=$(docker exec "$CONTAINER_NAME" mongosh --quiet --eval '
db = db.getSiblingDB("rag_server");
db.getCollectionNames().reduce((sum, col) => sum + db[col].countDocuments(), 0)
' 2>/dev/null || echo "0")

log_info "Documents before restore: $BEFORE_COUNT"

# Perform restore
log_info "Restoring from backup (this may take a moment)..."
docker exec -i "$CONTAINER_NAME" mongorestore --archive --drop < "$BACKUP_FILE"

if [ $? -ne 0 ]; then
    log_error "Restore failed!"
    exit 1
fi

# Get document count after restore
AFTER_COUNT=$(docker exec "$CONTAINER_NAME" mongosh --quiet --eval '
db = db.getSiblingDB("rag_server");
db.getCollectionNames().reduce((sum, col) => sum + db[col].countDocuments(), 0)
')

log_info "Documents after restore: $AFTER_COUNT"

# Show collection summary
log_info "Collection summary:"
docker exec "$CONTAINER_NAME" mongosh --quiet --eval '
db = db.getSiblingDB("rag_server");
db.getCollectionNames().forEach(col => {
  let count = db[col].countDocuments();
  if (count > 0) print("  " + col + ": " + count);
});
'

log_info "Restore completed successfully!"
