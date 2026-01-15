# Backup Documentation Index

This directory contains documentation for all backup systems configured on the LLM server.

## Backup Summary

| Backup | What | Schedule | Retention | Location |
|--------|------|----------|-----------|----------|
| [rsnapshot](Rsnapshot.md) | System files, projects, configs | Daily 2 AM | 7d/4w/3m | Internal SSD |
| [MongoDB](MongoDB.md) | All MongoDB databases | Daily 2 AM | 7 days | USB drive |

---

## Documents

### [Rsnapshot.md](Rsnapshot.md)

System backup using rsnapshot with hard-link-based rotating snapshots.

- **Backs up:** `/data/projects/`, `/home/chad/`, `/etc/`
- **Size:** ~19 GB
- **Storage:** `/data/backups/rsnapshot/` (internal SSD, 295 GB)
- **Key topics:** Configuration, exclusions, retention, restoration, troubleshooting

### [MongoDB.md](MongoDB.md)

MongoDB database backup using mongodump to external USB storage.

- **Backs up:** All databases (rag_server collections)
- **Size:** ~114 MB compressed
- **Storage:** `/mnt/usb/mongodb_backups/` (USB drive, 56 GB)
- **Key topics:** mongodump/mongorestore commands, USB handling, collection-level restore

---

## Quick Reference

| Task | Command |
|------|---------|
| Run rsnapshot backup | `sudo rsnapshot daily` |
| Run MongoDB backup | `sudo /data/projects/llm_website/scripts/mongodb_backup_usb.sh` |
| Check rsnapshot log | `sudo tail -f /var/log/rsnapshot.log` |
| Check MongoDB log | `sudo tail -f /var/log/mongodb_backup.log` |
| List rsnapshot backups | `ls /data/backups/rsnapshot/` |
| List MongoDB backups | `ls /mnt/usb/mongodb_backups/` |

---

## Storage Overview

```
Internal SSD (sdb4)              USB Drive (sdd1)
/data/backups/                   /mnt/usb/
└── rsnapshot/                   └── mongodb_backups/
    ├── daily.0-6/                   ├── backup_YYYYMMDD_HHMMSS/
    ├── weekly.0-3/                  └── LATEST.txt
    └── monthly.0-2/

Capacity: 295 GB                 Capacity: 56 GB
```

---

## Restoration Quick Reference

### Restore Files from rsnapshot

```bash
# Restore a single file
sudo cp /data/backups/rsnapshot/daily.0/localhost/data/projects/myfile.txt /data/projects/

# Restore a directory
sudo rsync -av /data/backups/rsnapshot/daily.0/localhost/data/projects/myproject/ /data/projects/myproject/

# Restore system config files
sudo cp -p /data/backups/rsnapshot/daily.0/localhost/etc/systemd/system/llama-sql.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### Restore MongoDB

```bash
# Find latest backup
cat /mnt/usb/mongodb_backups/LATEST.txt

# Restore entire backup
mongorestore --host=10.101.20.29 --port=27018 --gzip \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/

# Restore specific database
mongorestore --host=10.101.20.29 --port=27018 --gzip --db=rag_server \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/rag_server/

# Restore with drop (replace existing)
mongorestore --host=10.101.20.29 --port=27018 --gzip --drop \
    /mnt/usb/mongodb_backups/backup_YYYYMMDD_HHMMSS/
```

---

## Monitoring

### Check Backup Status

```bash
# rsnapshot log
sudo tail -50 /var/log/rsnapshot.log

# MongoDB log
sudo tail -50 /var/log/mongodb_backup.log

# Check for errors
sudo grep -i error /var/log/rsnapshot.log
sudo grep -i error /var/log/mongodb_backup.log

# Disk usage
sudo du -sh /data/backups/rsnapshot/*/
du -sh /mnt/usb/mongodb_backups/backup_*/

# Disk space
df -h /data/backups /mnt/usb
```

### Backup Health Check Script

Create `/usr/local/bin/check_all_backups.sh`:

```bash
#!/bin/bash
echo "=== BACKUP STATUS REPORT ==="
echo "Date: $(date)"
echo ""

echo "=== rsnapshot Backups ==="
if [ -d /data/backups/rsnapshot/daily.0 ]; then
    AGE=$(( ($(date +%s) - $(stat -c %Y /data/backups/rsnapshot/daily.0)) / 3600 ))
    echo "Latest daily: ${AGE} hours old"
    if [ $AGE -gt 48 ]; then
        echo "WARNING: Backup is more than 48 hours old!"
    fi
    sudo du -sh /data/backups/rsnapshot/daily.0/
else
    echo "ERROR: No rsnapshot backup found!"
fi

echo ""
echo "=== MongoDB Backups ==="
if [ -f /mnt/usb/mongodb_backups/LATEST.txt ]; then
    LATEST=$(cat /mnt/usb/mongodb_backups/LATEST.txt)
    echo "Latest: $LATEST"
    du -sh "$LATEST" 2>/dev/null || echo "Cannot read backup size"
else
    echo "WARNING: No MongoDB backup found or USB not mounted"
fi

echo ""
echo "=== Disk Space ==="
df -h /data/backups /mnt/usb 2>/dev/null

echo ""
echo "=== Backup Counts ==="
echo "rsnapshot snapshots: $(ls -1d /data/backups/rsnapshot/*/ 2>/dev/null | wc -l)"
echo "MongoDB backups: $(ls -1d /mnt/usb/mongodb_backups/backup_* 2>/dev/null | wc -l)"
```

Make executable:

```bash
sudo chmod +x /usr/local/bin/check_all_backups.sh
```

Run manually or add to cron for daily reports:

```bash
# Run manually
sudo /usr/local/bin/check_all_backups.sh

# Add to cron for daily 8 AM report
# 0 8 * * * /usr/local/bin/check_all_backups.sh >> /var/log/backup_status.log
```

---

## Troubleshooting

### rsnapshot Issues

| Problem | Solution |
|---------|----------|
| Syntax error in config | Use **tabs** not spaces. Run `sudo rsnapshot configtest` |
| Backup too slow | Add exclusions to `/etc/rsnapshot.conf` |
| Disk full | Reduce retention or add exclusions |
| Permission denied | Run as root: `sudo rsnapshot daily` |
| Lockfile exists | Check if running: `ps aux \| grep rsnapshot` |

### MongoDB Backup Issues

| Problem | Solution |
|---------|----------|
| USB not mounted | `sudo mount /dev/sdd1 /mnt/usb` |
| USB device not found | Check drive connected: `lsblk` |
| Insufficient space | Remove old backups or use larger drive |
| mongodump fails | Check MongoDB running: `systemctl status mongod` |
| Connection refused | Verify: `mongo --host 10.101.20.29 --port 27018` |

### Common Diagnostic Commands

```bash
# Check rsnapshot config
sudo rsnapshot configtest

# Test rsnapshot (dry run)
sudo rsnapshot -t daily

# Check MongoDB connection
mongo --host 10.101.20.29 --port 27018 --eval "db.stats()"

# Check USB mount
mountpoint /mnt/usb && echo "Mounted" || echo "Not mounted"

# Check cron jobs
sudo crontab -l | grep -E "rsnapshot|mongo"
cat /etc/cron.d/rsnapshot
```
