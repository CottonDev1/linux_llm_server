# Active Directory Integration Guide

This guide covers the Ubuntu LLM server's integration with the EWRINC.COM Active Directory domain, including AD authentication and Windows share mounting.

## Overview

| Setting | Value |
|---------|-------|
| **Server** | 10.101.20.21 |
| **Domain** | EWRINC.COM |
| **Client Software** | SSSD |
| **Join Method** | Realmd |

## Verify Domain Status

```bash
# Check domain join status
ssh chad@10.101.20.21 "realm list"

# Test AD user resolution
ssh chad@10.101.20.21 "id chad.walker@ewrinc"

# Check Kerberos tickets
ssh chad@10.101.20.21 "klist"
```

## AD User Login

Domain users can SSH to the server using their AD credentials:

```bash
# Login with AD username
ssh chad.walker@ewrinc@10.101.20.21

# Or with just username (if login-formats configured)
ssh chad.walker@10.101.20.21
```

## Managing Domain Access

```bash
# Allow specific user
ssh chad@10.101.20.21 "sudo realm permit chad.walker@ewrinc.com"

# Allow a group
ssh chad@10.101.20.21 "sudo realm permit -g 'linux_server_admins@ewrinc.com'"

# Allow all domain users
ssh chad@10.101.20.21 "sudo realm permit --all"

# Deny a user
ssh chad@10.101.20.21 "sudo realm deny chad.walker@ewrinc.com"
```

---

## Windows Share Mounting

### Target Share

| Setting | Value |
|---------|-------|
| **Server** | ewrtnfile1 |
| **Share** | TNShare |
| **UNC Path** | `\\ewrtnfile1\TNShare` |
| **Mount Point** | `/mnt/tnshare` |

---

## Method 1: Kerberos Authentication (Recommended)

Uses domain tickets - more secure, no stored passwords.

### Prerequisites

- Server must be domain-joined
- `cifs-utils` package installed
- Valid Kerberos ticket

### Step 1: Obtain Kerberos Ticket

```bash
# Get a Kerberos ticket (will prompt for password)
ssh chad@10.101.20.21 "kinit chad.walker@EWRINC.COM"

# Verify ticket
ssh chad@10.101.20.21 "klist"

# Example output:
# Ticket cache: FILE:/tmp/krb5cc_1000
# Default principal: chad.walker@EWRINC.COM
# Valid starting     Expires            Service principal
# 01/15/26 18:00:00  01/16/26 04:00:00  krbtgt/EWRINC.COM@EWRINC.COM
```

### Step 2: Create Mount Point

```bash
ssh chad@10.101.20.21 "sudo mkdir -p /mnt/tnshare"
```

### Step 3: Mount the Share

```bash
# Mount with Kerberos authentication
ssh chad@10.101.20.21 "sudo mount -t cifs //ewrtnfile1/TNShare /mnt/tnshare -o sec=krb5,cruid=$(id -u),vers=3.0"

# Verify mount
ssh chad@10.101.20.21 "df -h /mnt/tnshare"
ssh chad@10.101.20.21 "ls -la /mnt/tnshare"
```

### Mount Options Explained

| Option | Description |
|--------|-------------|
| `sec=krb5` | Use Kerberos authentication |
| `cruid=$(id -u)` | Use current user's Kerberos ticket |
| `vers=3.0` | SMB protocol version (use 3.0 or 2.1) |
| `uid=1000` | Set owner of mounted files (optional) |
| `gid=1000` | Set group of mounted files (optional) |
| `file_mode=0644` | Set file permissions (optional) |
| `dir_mode=0755` | Set directory permissions (optional) |

### Persistent Mount (fstab)

For automatic mounting at boot with Kerberos:

```bash
# Add to /etc/fstab
//ewrtnfile1/TNShare  /mnt/tnshare  cifs  sec=krb5,multiuser,vers=3.0,noauto,x-systemd.automount  0  0
```

**Note:** With `multiuser` and `x-systemd.automount`, each user accesses with their own Kerberos ticket.

### Kerberos Quick Reference

```bash
# Get Kerberos ticket
kinit chad.walker@EWRINC.COM

# Check ticket status
klist

# Mount share
sudo mount -t cifs //ewrtnfile1/TNShare /mnt/tnshare -o sec=krb5,cruid=$(id -u),vers=3.0

# Unmount share
sudo umount /mnt/tnshare

# Destroy ticket (logout)
kdestroy
```

---

## Method 2: Credentials File

Use when Kerberos isn't available or for service accounts.

### Step 1: Create Credentials File

```bash
# Create secure credentials file
ssh chad@10.101.20.21 "sudo bash -c 'cat > /root/.smbcredentials << EOF
username=svc_llmserver
password=YourSecurePassword
domain=EWRINC
EOF'"

# Secure the file
ssh chad@10.101.20.21 "sudo chmod 600 /root/.smbcredentials"
```

### Step 2: Mount the Share

```bash
# Create mount point
ssh chad@10.101.20.21 "sudo mkdir -p /mnt/tnshare"

# Mount with credentials file
ssh chad@10.101.20.21 "sudo mount -t cifs //ewrtnfile1/TNShare /mnt/tnshare -o credentials=/root/.smbcredentials,vers=3.0"
```

### Persistent Mount (fstab)

```bash
# Add to /etc/fstab
//ewrtnfile1/TNShare  /mnt/tnshare  cifs  credentials=/root/.smbcredentials,vers=3.0,uid=1000,gid=1000  0  0
```

---

## Automating Kerberos Ticket Renewal

For automated access (like backups), set up keytab-based authentication.

### Creating a Keytab

A keytab allows authentication without entering a password.

**Option 1: On Windows Domain Controller (PowerShell as Admin)**
```powershell
# Create keytab for service account
ktpass -princ svc_llmserver@EWRINC.COM -mapuser EWRINC\svc_llmserver -pass "ServiceAccountPassword" -crypto AES256-SHA1 -ptype KRB5_NT_PRINCIPAL -out svc_llmserver.keytab
```

**Option 2: Using ktutil on Linux**
```bash
ktutil
addent -password -p svc_llmserver@EWRINC.COM -k 1 -e aes256-cts-hmac-sha1-96
# Enter password when prompted
wkt /etc/krb5.keytab
quit
```

### Using the Keytab

```bash
# Copy keytab to server (if created on Windows)
scp svc_llmserver.keytab chad@10.101.20.21:/tmp/

# Install keytab
ssh chad@10.101.20.21 "sudo mv /tmp/svc_llmserver.keytab /etc/krb5.keytab && sudo chmod 600 /etc/krb5.keytab"

# Test keytab authentication
ssh chad@10.101.20.21 "sudo kinit -kt /etc/krb5.keytab svc_llmserver@EWRINC.COM && klist"
```

### Automatic Ticket Renewal (Cron)

```bash
# Add to root crontab for automatic renewal every 8 hours
ssh chad@10.101.20.21 "echo '0 */8 * * * kinit -kt /etc/krb5.keytab svc_llmserver@EWRINC.COM' | sudo tee -a /var/spool/cron/crontabs/root"
```

---

## Troubleshooting

### Domain Join Issues

| Problem | Solution |
|---------|----------|
| "realm: No such realm found" | Check DNS, ensure domain controller is reachable |
| "realm: Couldn't join realm" | Verify admin credentials, check time sync |
| AD users can't login | Run `realm permit --all` or permit specific users |

### Kerberos Issues

| Problem | Solution |
|---------|----------|
| "kinit: KDC reply did not match expectations" | Check domain name is uppercase (EWRINC.COM) |
| "Clock skew too great" | Sync time: `sudo ntpdate ewrinc.com` |
| "kinit: Cannot find KDC" | Check `/etc/krb5.conf`, verify DNS resolution |

### Share Mount Issues

| Problem | Solution |
|---------|----------|
| "mount error(126): Required key not available" | Run `kinit` to get a Kerberos ticket |
| "mount error(13): Permission denied" | Check share permissions, verify username |
| "mount error(112): Host is down" | Check server name resolution, try IP address |
| "mount error(2): No such file or directory" | Verify share name is correct |
| "mount error(95): Operation not supported" | Try different SMB version: `vers=2.1` or `vers=3.0` |
| Ticket expired | Run `kinit` again to renew ticket |

### Checking Logs

```bash
# SSSD logs (AD/authentication issues)
ssh chad@10.101.20.21 "sudo tail -50 /var/log/sssd/sssd_EWRINC.COM.log"

# Kerberos logs
ssh chad@10.101.20.21 "sudo tail -50 /var/log/auth.log | grep -i krb"

# Mount/CIFS logs
ssh chad@10.101.20.21 "dmesg | grep -i cifs | tail -20"
```

---

## Service Account Requirements

For automated share access, create a service account with:

| Requirement | Setting |
|-------------|---------|
| **Account type** | Regular domain user |
| **Password expiry** | Never expires |
| **Share permissions** | Read (or Modify if writing) on TNShare |
| **AD permissions** | None required (standard user) |

**Recommended naming:** `svc_llmserver` or `svc_llmbackup`
