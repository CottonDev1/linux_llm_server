# Vim Guide

A comprehensive guide to using Vim text editor on Ubuntu Server.

## Table of Contents

- [What is Vim?](#what-is-vim)
- [Installation](#installation)
- [Basic Concepts](#basic-concepts)
- [Modes](#modes)
- [Essential Commands](#essential-commands)
  - [Opening and Closing Files](#opening-and-closing-files)
  - [Navigation](#navigation)
  - [Editing](#editing)
  - [Copy, Cut, and Paste](#copy-cut-and-paste)
  - [Search and Replace](#search-and-replace)
  - [Undo and Redo](#undo-and-redo)
- [Working with Multiple Files](#working-with-multiple-files)
  - [Buffers](#buffers)
  - [Windows](#windows)
  - [Tabs](#tabs)
- [Visual Mode](#visual-mode)
- [Useful Commands](#useful-commands)
- [Configuration](#configuration)
- [Common Workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Quick Reference](#quick-reference)

---

## What is Vim?

Vim (Vi IMproved) is a highly configurable text editor built for efficient text editing. It's:

- Available on virtually every Unix/Linux system
- Extremely powerful once learned
- Highly customizable
- Efficient for keyboard-only editing (no mouse required)

---

## Installation

### Ubuntu/Debian

```bash
# Minimal vim
sudo apt update
sudo apt install vim

# Full-featured vim with GUI support
sudo apt install vim-gtk3
```

### Verify Installation

```bash
vim --version | head -1
```

---

## Basic Concepts

Vim is a **modal editor**, meaning it has different modes for different tasks:

- **Normal mode** - Navigate and execute commands (default)
- **Insert mode** - Type and edit text
- **Visual mode** - Select text
- **Command-line mode** - Execute extended commands

The key to learning Vim is understanding these modes and how to switch between them.

---

## Modes

### Normal Mode (Default)

This is the default mode when you open Vim. Used for navigation and commands.

- Press `Esc` from any mode to return to Normal mode

### Insert Mode

Used for typing text. Enter Insert mode with:

| Key | Action |
|-----|--------|
| `i` | Insert before cursor |
| `I` | Insert at beginning of line |
| `a` | Append after cursor |
| `A` | Append at end of line |
| `o` | Open new line below |
| `O` | Open new line above |

Press `Esc` to return to Normal mode.

### Visual Mode

Used for selecting text:

| Key | Action |
|-----|--------|
| `v` | Character-wise selection |
| `V` | Line-wise selection |
| `Ctrl+v` | Block (column) selection |

### Command-line Mode

Enter with `:` from Normal mode. Used for saving, quitting, searching, etc.

---

## Essential Commands

### Opening and Closing Files

**Open a file:**
```bash
vim filename
```

**Open at specific line:**
```bash
vim +42 filename    # Open at line 42
vim +/pattern file  # Open at first match of pattern
```

**Save and quit (from Normal mode):**

| Command | Action |
|---------|--------|
| `:w` | Save (write) |
| `:q` | Quit |
| `:wq` or `:x` | Save and quit |
| `:q!` | Quit without saving |
| `ZZ` | Save and quit (shortcut) |
| `ZQ` | Quit without saving (shortcut) |

### Navigation

**Basic movement:**

| Key | Action |
|-----|--------|
| `h` | Left |
| `j` | Down |
| `k` | Up |
| `l` | Right |
| Arrow keys | Also work |

**Word movement:**

| Key | Action |
|-----|--------|
| `w` | Next word (beginning) |
| `W` | Next WORD (space-delimited) |
| `e` | End of word |
| `b` | Previous word |
| `B` | Previous WORD |

**Line movement:**

| Key | Action |
|-----|--------|
| `0` | Beginning of line |
| `^` | First non-blank character |
| `$` | End of line |
| `g_` | Last non-blank character |

**Screen movement:**

| Key | Action |
|-----|--------|
| `gg` | Go to first line |
| `G` | Go to last line |
| `42G` or `:42` | Go to line 42 |
| `Ctrl+f` | Page forward |
| `Ctrl+b` | Page backward |
| `Ctrl+d` | Half page down |
| `Ctrl+u` | Half page up |
| `H` | Top of screen |
| `M` | Middle of screen |
| `L` | Bottom of screen |

**Jump movement:**

| Key | Action |
|-----|--------|
| `%` | Jump to matching bracket |
| `{` | Previous paragraph |
| `}` | Next paragraph |
| `Ctrl+o` | Jump back |
| `Ctrl+i` | Jump forward |

### Editing

**Delete:**

| Key | Action |
|-----|--------|
| `x` | Delete character under cursor |
| `X` | Delete character before cursor |
| `dw` | Delete word |
| `dd` | Delete line |
| `d$` or `D` | Delete to end of line |
| `d0` | Delete to beginning of line |
| `dgg` | Delete to beginning of file |
| `dG` | Delete to end of file |

**Change (delete and enter Insert mode):**

| Key | Action |
|-----|--------|
| `cw` | Change word |
| `cc` | Change entire line |
| `c$` or `C` | Change to end of line |
| `ci"` | Change inside quotes |
| `ci(` | Change inside parentheses |
| `ca{` | Change around braces (including braces) |

**Other editing:**

| Key | Action |
|-----|--------|
| `r` | Replace single character |
| `R` | Enter Replace mode |
| `J` | Join current line with next |
| `~` | Toggle case |
| `>>` | Indent line |
| `<<` | Unindent line |
| `.` | Repeat last command |

### Copy, Cut, and Paste

Vim uses "yank" for copy and "delete" doubles as cut.

| Key | Action |
|-----|--------|
| `yy` or `Y` | Yank (copy) line |
| `yw` | Yank word |
| `y$` | Yank to end of line |
| `p` | Paste after cursor |
| `P` | Paste before cursor |
| `dd` | Cut line (delete stores in register) |
| `"ay` | Yank to register 'a' |
| `"ap` | Paste from register 'a' |
| `"+y` | Yank to system clipboard |
| `"+p` | Paste from system clipboard |

### Search and Replace

**Search:**

| Key | Action |
|-----|--------|
| `/pattern` | Search forward |
| `?pattern` | Search backward |
| `n` | Next match |
| `N` | Previous match |
| `*` | Search word under cursor (forward) |
| `#` | Search word under cursor (backward) |

**Replace:**

```vim
:s/old/new/          " Replace first occurrence on current line
:s/old/new/g         " Replace all on current line
:%s/old/new/g        " Replace all in file
:%s/old/new/gc       " Replace all with confirmation
:5,10s/old/new/g     " Replace in lines 5-10
```

### Undo and Redo

| Key | Action |
|-----|--------|
| `u` | Undo |
| `Ctrl+r` | Redo |
| `U` | Undo all changes on current line |

---

## Working with Multiple Files

### Buffers

Buffers are in-memory text files.

| Command | Action |
|---------|--------|
| `:e filename` | Edit file in new buffer |
| `:ls` or `:buffers` | List buffers |
| `:b N` | Switch to buffer N |
| `:bn` | Next buffer |
| `:bp` | Previous buffer |
| `:bd` | Delete (close) buffer |
| `:b name` | Switch to buffer by name |

### Windows

Split the screen to view multiple buffers.

| Command | Action |
|---------|--------|
| `:split` or `:sp` | Horizontal split |
| `:vsplit` or `:vs` | Vertical split |
| `:split file` | Split and open file |
| `Ctrl+w s` | Horizontal split |
| `Ctrl+w v` | Vertical split |
| `Ctrl+w w` | Cycle through windows |
| `Ctrl+w h/j/k/l` | Navigate windows |
| `Ctrl+w q` | Close window |
| `Ctrl+w o` | Close all other windows |
| `Ctrl+w =` | Equal size windows |
| `Ctrl+w _` | Maximize height |
| `Ctrl+w \|` | Maximize width |

### Tabs

| Command | Action |
|---------|--------|
| `:tabnew` | New tab |
| `:tabnew file` | New tab with file |
| `:tabn` or `gt` | Next tab |
| `:tabp` or `gT` | Previous tab |
| `:tabclose` | Close tab |
| `:tabs` | List tabs |
| `Ngt` | Go to tab N |

---

## Visual Mode

### Selecting Text

1. Press `v`, `V`, or `Ctrl+v` to enter Visual mode
2. Use movement keys to select
3. Perform action on selection

### Visual Mode Actions

| Key | Action |
|-----|--------|
| `d` | Delete selection |
| `y` | Yank selection |
| `c` | Change selection |
| `>` | Indent |
| `<` | Unindent |
| `u` | Lowercase |
| `U` | Uppercase |
| `~` | Toggle case |
| `:` | Command on selection |

### Block Selection (Ctrl+v)

Useful for editing columns:

1. `Ctrl+v` to enter block visual mode
2. Select column
3. `I` to insert at beginning of each line
4. Type text
5. `Esc` to apply to all lines

---

## Useful Commands

### File Operations

| Command | Action |
|---------|--------|
| `:w newname` | Save as new file |
| `:r filename` | Insert file contents |
| `:r !command` | Insert command output |
| `:e!` | Reload file (discard changes) |
| `:pwd` | Print working directory |
| `:cd path` | Change directory |

### Information

| Command | Action |
|---------|--------|
| `Ctrl+g` | Show file info |
| `g Ctrl+g` | Word/character count |
| `:set number` | Show line numbers |
| `:set relativenumber` | Relative line numbers |
| `:set list` | Show whitespace |

### Marks

| Command | Action |
|---------|--------|
| `ma` | Set mark 'a' |
| `'a` | Jump to line of mark 'a' |
| `` `a `` | Jump to exact position of mark 'a' |
| `:marks` | List marks |

### Macros

| Command | Action |
|---------|--------|
| `qa` | Start recording macro 'a' |
| `q` | Stop recording |
| `@a` | Play macro 'a' |
| `@@` | Replay last macro |
| `10@a` | Play macro 'a' 10 times |

### Text Objects

Use with operators (d, c, y, v):

| Object | Description |
|--------|-------------|
| `iw` | Inner word |
| `aw` | A word (includes space) |
| `i"` | Inside double quotes |
| `a"` | Around double quotes |
| `i(` or `ib` | Inside parentheses |
| `i{` or `iB` | Inside braces |
| `it` | Inside HTML tag |
| `ip` | Inside paragraph |

Examples:
- `ciw` - Change inner word
- `da"` - Delete around quotes
- `vi{` - Select inside braces

---

## Configuration

Vim configuration is stored in `~/.vimrc`.

### Sample Configuration

```vim
" Basic settings
set nocompatible          " Use Vim settings, not Vi
syntax on                 " Syntax highlighting
filetype plugin indent on " Filetype detection

" Interface
set number                " Line numbers
set relativenumber        " Relative line numbers
set ruler                 " Show cursor position
set showcmd               " Show incomplete commands
set showmode              " Show current mode
set wildmenu              " Command line completion
set laststatus=2          " Always show status line

" Search
set hlsearch              " Highlight search results
set incsearch             " Incremental search
set ignorecase            " Case insensitive search
set smartcase             " Unless uppercase used

" Indentation
set autoindent            " Auto indent
set smartindent           " Smart indent
set expandtab             " Spaces instead of tabs
set tabstop=4             " Tab width
set shiftwidth=4          " Indent width
set softtabstop=4         " Soft tab width

" Editing
set backspace=indent,eol,start  " Better backspace
set clipboard=unnamedplus       " System clipboard
set mouse=a                     " Mouse support
set scrolloff=5                 " Lines above/below cursor

" Visual
set cursorline            " Highlight current line
set colorcolumn=80        " Column marker
set wrap                  " Wrap lines
set linebreak             " Wrap at word boundaries

" Files
set encoding=utf-8        " UTF-8 encoding
set nobackup              " No backup files
set noswapfile            " No swap files
set autoread              " Auto reload changed files

" Key mappings
let mapleader = " "       " Space as leader key

" Quick save
nnoremap <leader>w :w<CR>

" Quick quit
nnoremap <leader>q :q<CR>

" Clear search highlight
nnoremap <leader>/ :noh<CR>

" Easy window navigation
nnoremap <C-h> <C-w>h
nnoremap <C-j> <C-w>j
nnoremap <C-k> <C-w>k
nnoremap <C-l> <C-w>l

" Move lines up/down
nnoremap <A-j> :m .+1<CR>==
nnoremap <A-k> :m .-2<CR>==
vnoremap <A-j> :m '>+1<CR>gv=gv
vnoremap <A-k> :m '<-2<CR>gv=gv
```

### Reload Configuration

```vim
:source ~/.vimrc
```

---

## Common Workflows

### Quick Edit and Save

```
vim file.txt    # Open file
i               # Enter Insert mode
[type text]
Esc             # Return to Normal mode
:wq             # Save and quit
```

### Search and Replace All

```
vim file.txt
:%s/oldtext/newtext/g
:wq
```

### Edit Multiple Files

```bash
vim file1.txt file2.txt
```

Then use `:n` for next file, `:prev` for previous.

### Compare Two Files

```bash
vim -d file1.txt file2.txt
```

### Edit Remote File

```bash
vim scp://user@host//path/to/file
```

### Record and Apply Macro

```
qa              # Start recording to register 'a'
[do actions]
q               # Stop recording
100@a           # Apply macro 100 times
```

---

## Troubleshooting

### Stuck in a Mode

Press `Esc` multiple times to return to Normal mode.

### Can't Quit

Try these in order:
1. `Esc` then `:q`
2. `Esc` then `:q!` (discard changes)
3. `Esc` then `:qa!` (quit all without saving)

### Swap File Exists

```
E325: ATTENTION - Found swap file
```

Options:
- `R` - Recover
- `D` - Delete swap and edit
- `Q` - Quit
- `A` - Abort

### Paste Causes Indent Issues

Before pasting:
```vim
:set paste
```

After pasting:
```vim
:set nopaste
```

### Arrow Keys Show A B C D

Add to `~/.vimrc`:
```vim
set nocompatible
```

---

## Quick Reference

### Movement

| Key | Action |
|-----|--------|
| `h j k l` | Left, down, up, right |
| `w / b` | Next/previous word |
| `0 / $` | Start/end of line |
| `gg / G` | Start/end of file |
| `Ctrl+f/b` | Page down/up |

### Mode Switching

| Key | Action |
|-----|--------|
| `i` | Insert mode |
| `v` | Visual mode |
| `Esc` | Normal mode |
| `:` | Command mode |

### Editing

| Key | Action |
|-----|--------|
| `x` | Delete character |
| `dd` | Delete line |
| `yy` | Copy line |
| `p` | Paste |
| `u` | Undo |
| `Ctrl+r` | Redo |

### Files

| Command | Action |
|---------|--------|
| `:w` | Save |
| `:q` | Quit |
| `:wq` | Save and quit |
| `:q!` | Quit without saving |
| `:e file` | Open file |

### Search

| Key | Action |
|-----|--------|
| `/text` | Search forward |
| `n / N` | Next/previous match |
| `:%s/a/b/g` | Replace all |

### Windows

| Key | Action |
|-----|--------|
| `:sp` | Horizontal split |
| `:vs` | Vertical split |
| `Ctrl+w w` | Switch window |
| `Ctrl+w q` | Close window |
