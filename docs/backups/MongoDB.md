# MongoDB Backup

MongoDB is backed up daily using `mongodump` to an external USB drive for offsite storage capability.

## Overview

| Setting | Value |
|---------|-------|
| **MongoDB Host** | 10.101.20.21 |
| **MongoDB Port** | 27017 |
| **Backup Location** | `/mnt/usb/mongodb_backups/` |
| **USB Device** | `/dev/sdd1` (WD Passport) |
| **Schedule** | Daily 2:00 AM |
| **Retention** | 7 days |
| **Compression** | gzip |
| **Typical Size** | ~114 MB |

---

## What's Backed Up

All databases on the MongoDB server are backed up, including:

| Database | Collection | Description |
|----------|------------|-------------|
| `rag_server` | `code_classes` | Code class definitions (1,121 docs) |
| `rag_server` | `code_methods` | Code method definitions (8,024 docs) |
| `rag_server` | `code_callgraph` | Code call relationships (28,281 docs) |
| `rag_server` | `query_sessions` | Query session data |
| `rag_server` | `phone_customer_map` | Phone-customer mappings |
| `rag_server` | `sql_knowledge` | SQL knowledge base |

---

## Backup Script

**Location:** `/data/projects/llm_website/scripts/mongodb_backup_usb.sh`

### What the Script Does

1. Checks if USB device is connected
2. Mounts USB drive if not already mounted
3. Verifies at least 1GB free space
4. Runs `mongodump` with gzip compression
5. Creates timestamped backup directory
6. Removes backups older than 7 days
7. Syncs filesystem

### Cron Entry

```cron
0 2 * * * /data/projects/llm_website/scripts/mongodb_backup_usb.sh >> /var/log/mongodb_backup.log 2>&1
```

---

## Directory Structure

```
/mnt/usb/mongodb_backups/
├── backup_20260115_020000/
│   └── rag_server/
│       ├── code_classes.bson.gz
│       ├── code_classes.metadata.json.gz
│       ├── code_methods.bson.gz
│       ├── code_methods.metadata.json.gz
│       ├── code_callgraph.bson.gz
│       └── ...
├── backup_20260114_020000/
├── backup_20260113_020000/
└── LATEST.txt              # Contains path to most recent backup
```

---

## Manual Commands

### Run Backup Manually

```bash
sudo /data/projects/llm_website/scripts/mongodb_backup_usb.sh
```

### Check Backup Log

```bash
sudo tail -50 /var/log/mongodb_backup.log
```

### List Available Backups

```bash
ls -la /mnt/usb/mongodb_backups/
```

### Check Latest Backup

```bash
cat /mnt/usb/mongodb_backups/LATEST.txt
```

### Check Backup Size

```bash
du -sh /mnt/usb/mongodb_backups/backup_*/
```

### Mount USB Drive Manually

```bash
sudo mount /dev/sdd1 /mnt/usb
```

---

## Restoration

### Restore Entire Backup

```bash
mongorestore \
    --host=10.101.20.21 \
    --port=27017 \
    --gzip \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/
```

### Restore Specific Database

```bash
mongorestore \
    --host=10.101.20.21 \
    --port=27017 \
    --gzip \
    --db=rag_server \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/rag_server/
```

### Restore Specific Collection

```bash
mongorestore \
    --host=10.101.20.21 \
    --port=27017 \
    --gzip \
    --db=rag_server \
    --collection=code_methods \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/rag_server/code_methods.bson.gz
```

### Restore with Drop (Replace Existing Data)

**Warning:** This will delete existing data before restoring!

```bash
mongorestore \
    --host=10.101.20.21 \
    --port=27017 \
    --gzip \
    --drop \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/
```

### Restore to Different Database

```bash
mongorestore \
    --host=10.101.20.21 \
    --port=27017 \
    --gzip \
    --db=rag_server_restored \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/rag_server/
```

---

## USB Drive Information

| Property | Value |
|----------|-------|
| **Device** | `/dev/sdd1` |
| **Mount Point** | `/mnt/usb` |
| **Filesystem** | FAT32 (vfat) |
| **Label** | WD Passport |
| **Capacity** | 56 GB |

**Note:** FAT32 doesn't support symlinks, so `LATEST.txt` file is used instead of a symlink.

### USB Drive Commands

```bash
# Check if mounted
mountpoint /mnt/usb

# Mount manually
sudo mount /dev/sdd1 /mnt/usb

# Unmount (safe removal)
sudo umount /mnt/usb

# Check USB device exists
lsblk | grep sdd
```

---

## Configuration

The backup script uses these settings (defined at top of script):

```bash
MONGO_HOST="10.101.20.21"
MONGO_PORT="27017"
USB_MOUNT="/mnt/usb"
USB_DEVICE="/dev/sdd1"
BACKUP_DIR="${USB_MOUNT}/mongodb_backups"
LOG_FILE="/var/log/mongodb_backup.log"
RETENTION_DAYS=7
```

To modify settings, edit the script:

```bash
sudo nano /data/projects/llm_website/scripts/mongodb_backup_usb.sh
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| USB not mounted | `sudo mount /dev/sdd1 /mnt/usb` |
| USB device not found | Check drive is connected: `lsblk` |
| Insufficient space | Remove old backups or use larger drive |
| mongodump fails | Check MongoDB is running on remote host |
| Connection refused | Verify host/port: `mongo --host 10.101.20.21 --port 27017` |
| Permission denied | Script must run as root |

### Verify MongoDB Connection

```bash
mongo --host 10.101.20.21 --port 27017 --eval "db.stats()"
```

### Check Script Permissions

```bash
ls -la /data/projects/llm_website/scripts/mongodb_backup_usb.sh
```

### View Full Backup Log

```bash
sudo cat /var/log/mongodb_backup.log
```
