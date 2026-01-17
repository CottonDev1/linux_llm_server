# Tmux Guide

A comprehensive guide to using tmux (terminal multiplexer) on Ubuntu Server.

## Table of Contents

- [What is Tmux?](#what-is-tmux)
- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Sessions](#sessions)
  - [Creating Sessions](#creating-sessions)
  - [Detaching and Attaching](#detaching-and-attaching)
  - [Listing and Killing Sessions](#listing-and-killing-sessions)
- [Windows](#windows)
  - [Creating Windows](#creating-windows)
  - [Navigating Windows](#navigating-windows)
  - [Managing Windows](#managing-windows)
- [Panes](#panes)
  - [Splitting Panes](#splitting-panes)
  - [Navigating Panes](#navigating-panes)
  - [Resizing Panes](#resizing-panes)
  - [Pane Layouts](#pane-layouts)
- [Copy Mode](#copy-mode)
- [Configuration](#configuration)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## What is Tmux?

Tmux is a terminal multiplexer that allows you to:

- Run multiple terminal sessions within a single window
- Keep processes running after disconnecting from SSH
- Split your terminal into multiple panes
- Share terminal sessions between multiple users

This is especially useful for headless Ubuntu servers where you need to manage multiple long-running processes.

---

## Installation

### Ubuntu/Debian

```bash
sudo apt update
sudo apt install tmux
```

### Verify Installation

```bash
tmux -V
```

Expected output: `tmux 3.x` (version may vary)

---

## Basic Concepts

Tmux uses a hierarchy of three main components:

1. **Session** - A collection of windows managed by tmux
2. **Window** - A single screen within a session (like browser tabs)
3. **Pane** - A subdivision of a window (split screens)

### The Prefix Key

Most tmux commands start with a **prefix key**, which is `Ctrl+b` by default.

**Notation:** `Ctrl+b` then `x` means:
1. Press `Ctrl` and `b` together
2. Release both keys
3. Press `x`

---

## Sessions

### Creating Sessions

**Start a new unnamed session:**
```bash
tmux
```

**Start a new named session:**
```bash
tmux new -s mysession
```

**Start a session with a specific window name:**
```bash
tmux new -s mysession -n mywindow
```

### Detaching and Attaching

**Detach from current session (keeps it running):**
```
Ctrl+b then d
```

**Attach to a session:**
```bash
tmux attach -t mysession
```

**Attach to last session:**
```bash
tmux attach
# or
tmux a
```

### Listing and Killing Sessions

**List all sessions:**
```bash
tmux ls
```

**Kill a specific session:**
```bash
tmux kill-session -t mysession
```

**Kill all sessions:**
```bash
tmux kill-server
```

**Rename current session:**
```
Ctrl+b then $
```

---

## Windows

### Creating Windows

**Create a new window:**
```
Ctrl+b then c
```

**Create a window with a name:**
```bash
# From command line when starting
tmux new -s mysession -n editor
```

### Navigating Windows

| Command | Action |
|---------|--------|
| `Ctrl+b` then `n` | Next window |
| `Ctrl+b` then `p` | Previous window |
| `Ctrl+b` then `0-9` | Go to window by number |
| `Ctrl+b` then `w` | Interactive window list |
| `Ctrl+b` then `l` | Last active window |

### Managing Windows

**Rename current window:**
```
Ctrl+b then ,
```

**Close current window:**
```
Ctrl+b then &
```
Or simply type `exit` in the shell.

**Move window:**
```
Ctrl+b then .
```
Then enter the new window number.

---

## Panes

### Splitting Panes

**Split vertically (side by side):**
```
Ctrl+b then %
```

**Split horizontally (top and bottom):**
```
Ctrl+b then "
```

### Navigating Panes

| Command | Action |
|---------|--------|
| `Ctrl+b` then `Arrow keys` | Move between panes |
| `Ctrl+b` then `o` | Cycle through panes |
| `Ctrl+b` then `q` | Show pane numbers (press number to jump) |
| `Ctrl+b` then `;` | Toggle to last active pane |

### Resizing Panes

**Using prefix + arrow keys:**
```
Ctrl+b then Ctrl+Arrow keys
```

**Resize in larger increments:**
```
Ctrl+b then Alt+Arrow keys
```

### Pane Layouts

**Cycle through preset layouts:**
```
Ctrl+b then Space
```

**Available layouts:**
- `even-horizontal` - Panes spread horizontally
- `even-vertical` - Panes spread vertically
- `main-horizontal` - One large pane on top, others below
- `main-vertical` - One large pane on left, others on right
- `tiled` - All panes equal size

**Select specific layout:**
```
Ctrl+b then Alt+1  # even-horizontal
Ctrl+b then Alt+2  # even-vertical
Ctrl+b then Alt+3  # main-horizontal
Ctrl+b then Alt+4  # main-vertical
Ctrl+b then Alt+5  # tiled
```

### Other Pane Commands

**Close current pane:**
```
Ctrl+b then x
```

**Zoom pane (toggle fullscreen):**
```
Ctrl+b then z
```

**Convert pane to window:**
```
Ctrl+b then !
```

**Swap panes:**
```
Ctrl+b then {  # Swap with previous
Ctrl+b then }  # Swap with next
```

---

## Copy Mode

Copy mode allows you to scroll through terminal output and copy text.

**Enter copy mode:**
```
Ctrl+b then [
```

**Navigation in copy mode:**
- Arrow keys or `h`, `j`, `k`, `l` (vim-style)
- `Ctrl+u` / `Ctrl+d` - Page up/down
- `g` / `G` - Go to top/bottom
- `/` - Search forward
- `?` - Search backward
- `n` / `N` - Next/previous search result

**Copy text:**
1. Enter copy mode: `Ctrl+b` then `[`
2. Navigate to start of text
3. Press `Space` to start selection
4. Navigate to end of text
5. Press `Enter` to copy

**Paste text:**
```
Ctrl+b then ]
```

**Exit copy mode:**
```
q or Escape
```

---

## Configuration

Tmux configuration is stored in `~/.tmux.conf`.

### Sample Configuration

Create or edit `~/.tmux.conf`:

```bash
# Change prefix from Ctrl+b to Ctrl+a
# set -g prefix C-a
# unbind C-b
# bind C-a send-prefix

# Enable mouse support
set -g mouse on

# Start window numbering at 1
set -g base-index 1
setw -g pane-base-index 1

# Renumber windows when one is closed
set -g renumber-windows on

# Increase scrollback buffer
set -g history-limit 10000

# Faster key repetition
set -s escape-time 0

# Use vi keys in copy mode
setw -g mode-keys vi

# Split panes using | and -
bind | split-window -h
bind - split-window -v

# Reload config
bind r source-file ~/.tmux.conf \; display "Config reloaded!"

# Easy pane navigation
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# Status bar customization
set -g status-style 'bg=#333333 fg=#ffffff'
set -g status-left-length 40
set -g status-left '#[fg=green]Session: #S #[fg=yellow]Window: #I #[fg=cyan]Pane: #P'
set -g status-right '#[fg=cyan]%Y-%m-%d %H:%M'
```

**Reload configuration without restarting:**
```
Ctrl+b then :
source-file ~/.tmux.conf
```

Or if you added the binding above:
```
Ctrl+b then r
```

---

## Common Workflows

### Development Environment

```bash
# Create a session for development
tmux new -s dev

# Create windows for different tasks
Ctrl+b c  # Window 1: Editor
Ctrl+b c  # Window 2: Server
Ctrl+b c  # Window 3: Git/Tests

# Rename windows
Ctrl+b ,  # Type "editor"
Ctrl+b n  # Next window
Ctrl+b ,  # Type "server"
# etc.
```

### Running Long Processes

```bash
# Start tmux session
tmux new -s backup

# Run your long process
./run-backup.sh

# Detach (process keeps running)
Ctrl+b d

# Later, reattach to check progress
tmux attach -t backup
```

### Monitoring Multiple Services

```bash
# Create session
tmux new -s monitor

# Split into 4 panes
Ctrl+b %      # Split vertical
Ctrl+b "      # Split horizontal
Ctrl+b o      # Move to next pane
Ctrl+b "      # Split horizontal

# Run different monitors in each pane
# Pane 1: htop
# Pane 2: tail -f /var/log/syslog
# Pane 3: watch df -h
# Pane 4: journalctl -f
```

---

## Troubleshooting

### Session won't attach
```bash
# List sessions to verify name
tmux ls

# Kill problematic session and restart
tmux kill-session -t problemsession
```

### Terminal colors look wrong
Add to `~/.tmux.conf`:
```bash
set -g default-terminal "screen-256color"
```

### Mouse not working
Add to `~/.tmux.conf`:
```bash
set -g mouse on
```

### Nested tmux sessions
If you SSH into a server that's also running tmux, use a different prefix or send prefix twice:
```
Ctrl+b Ctrl+b [command]  # Sends command to inner tmux
```

### Clear tmux scrollback
```
Ctrl+b then :
clear-history
```

---

## Quick Reference

### Essential Commands

| Action | Command |
|--------|---------|
| **Sessions** | |
| New session | `tmux new -s name` |
| List sessions | `tmux ls` |
| Attach | `tmux attach -t name` |
| Detach | `Ctrl+b d` |
| Kill session | `tmux kill-session -t name` |
| **Windows** | |
| New window | `Ctrl+b c` |
| Next window | `Ctrl+b n` |
| Previous window | `Ctrl+b p` |
| Window list | `Ctrl+b w` |
| Rename window | `Ctrl+b ,` |
| Close window | `Ctrl+b &` |
| **Panes** | |
| Split vertical | `Ctrl+b %` |
| Split horizontal | `Ctrl+b "` |
| Navigate | `Ctrl+b Arrow` |
| Zoom toggle | `Ctrl+b z` |
| Close pane | `Ctrl+b x` |
| **Other** | |
| Command mode | `Ctrl+b :` |
| Copy mode | `Ctrl+b [` |
| Paste | `Ctrl+b ]` |
| Help | `Ctrl+b ?` |

### Exiting Tmux

| Goal | Method |
|------|--------|
| Detach (keep running) | `Ctrl+b d` |
| Close current pane | `Ctrl+b x` or `exit` |
| Close current window | `Ctrl+b &` or `exit` all panes |
| Kill session | `tmux kill-session -t name` |
| Kill all sessions | `tmux kill-server` |
