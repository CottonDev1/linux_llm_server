# rsnapshot System Backup

rsnapshot creates rotating snapshots of system files using rsync and hard links. Each snapshot appears as a full backup but only changed files consume additional disk space.

## Overview

| Setting | Value |
|---------|-------|
| **Backup Location** | `/data/backups/rsnapshot/` |
| **Schedule** | Daily 2:00 AM |
| **Total Size** | ~19 GB |
| **Config File** | `/etc/rsnapshot.conf` |
| **Log File** | `/var/log/rsnapshot.log` |

---

## What's Backed Up

| Directory | Description | Approx Size |
|-----------|-------------|-------------|
| `/data/projects/` | LLM server code, llama.cpp, configs | ~18 GB |
| `/home/chad/` | User home directory | ~500 MB |
| `/etc/` | System configuration files | ~50 MB |

### Exclusions

The following are excluded to save space:

- `*.log` - Log files
- `*.tmp` - Temporary files
- `__pycache__/` - Python cache
- `node_modules/` - Node.js dependencies
- `.cache/` - Cache directories
- `*.pyc` - Compiled Python
- `.git/objects/` - Git object database
- `*.gguf` - LLM model files (can be re-downloaded)

---

## Retention Schedule

| Interval | Time | Retention | Cron Entry |
|----------|------|-----------|------------|
| Daily | 2:00 AM | 7 snapshots | `0 2 * * *` |
| Weekly | 3:00 AM Sunday | 4 snapshots | `0 3 * * 0` |
| Monthly | 4:00 AM 1st | 3 snapshots | `0 4 1 * *` |

---

## Directory Structure

```
/data/backups/rsnapshot/
├── daily.0/          # Most recent daily (yesterday)
├── daily.1/          # 2 days ago
├── daily.2/          # 3 days ago
├── ...
├── daily.6/          # 7 days ago
├── weekly.0/         # Most recent weekly
├── weekly.1/         # 2 weeks ago
├── ...
├── monthly.0/        # Most recent monthly
└── monthly.2/        # 3 months ago
```

Each snapshot contains:
```
daily.0/localhost/
├── data/projects/    # LLM server projects
├── etc/              # System configuration
└── home/chad/        # User home directory
```

---

## How Hard Links Work

rsnapshot uses hard links for unchanged files, making each snapshot look like a full backup while only using space for changed files:

```
daily.0/                    daily.1/
├── file1.txt (changed)     ├── file1.txt (old version)
├── file2.txt ─────────────►├── file2.txt (same inode)
└── file3.txt ─────────────►└── file3.txt (same inode)
```

If 19GB of data has only 100MB of daily changes, the second backup uses ~100MB additional space.

---

## Manual Commands

```bash
# Run daily backup manually
sudo rsnapshot daily

# Run weekly backup
sudo rsnapshot weekly

# Run monthly backup
sudo rsnapshot monthly

# Dry run (test without making changes)
sudo rsnapshot -t daily

# Test configuration syntax
sudo rsnapshot configtest

# View backup log
sudo tail -f /var/log/rsnapshot.log

# Check backup sizes
sudo du -sh /data/backups/rsnapshot/*/

# Check disk space
df -h /data/backups
```

---

## Restoration

### Restore a Single File

```bash
sudo cp /data/backups/rsnapshot/daily.0/localhost/data/projects/myfile.txt /data/projects/
```

### Restore a Directory

```bash
sudo rsync -av /data/backups/rsnapshot/daily.0/localhost/data/projects/myproject/ /data/projects/myproject/
```

### Restore System Config Files

```bash
sudo cp -p /data/backups/rsnapshot/daily.0/localhost/etc/systemd/system/llama-sql.service /etc/systemd/system/
sudo systemctl daemon-reload
```

### Restore to Different Location (for comparison)

```bash
sudo rsync -av /data/backups/rsnapshot/daily.1/localhost/data/projects/ /tmp/restored_projects/
```

---

## Configuration

### Main Config: `/etc/rsnapshot.conf`

```conf
config_version	1.2
snapshot_root	/data/backups/rsnapshot/

cmd_cp		/bin/cp
cmd_rm		/bin/rm
cmd_rsync	/usr/bin/rsync
cmd_logger	/usr/bin/logger

retain	daily	7
retain	weekly	4
retain	monthly	3

verbose		2
loglevel	3
logfile	/var/log/rsnapshot.log
lockfile	/var/run/rsnapshot.pid

exclude	*.log
exclude	*.tmp
exclude	__pycache__/
exclude	node_modules/
exclude	.cache/
exclude	*.pyc
exclude	.git/objects/
exclude	*.gguf

backup	/data/projects/	localhost/
backup	/home/chad/	localhost/
backup	/etc/	localhost/
```

**Important:** This file requires **tabs** between parameters, not spaces!

### Cron Config: `/etc/cron.d/rsnapshot`

```cron
# Daily backup at 2:00 AM
0 2 * * * root /usr/bin/rsnapshot daily

# Weekly backup at 3:00 AM on Sundays
0 3 * * 0 root /usr/bin/rsnapshot weekly

# Monthly backup at 4:00 AM on the 1st
0 4 1 * * root /usr/bin/rsnapshot monthly
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Syntax error in config | Use tabs, not spaces. Run `sudo rsnapshot configtest` |
| Backup too slow | Add more exclusions to config |
| Disk full | Reduce retention or add exclusions |
| Permission denied | Run as root: `sudo rsnapshot daily` |
| Lockfile exists | Check if backup is running: `ps aux \| grep rsnapshot` |

### Check for Errors

```bash
sudo grep -i error /var/log/rsnapshot.log
```

### Verify Backup Age

```bash
stat /data/backups/rsnapshot/daily.0/
```
