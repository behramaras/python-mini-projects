````markdown
# Tic-Tac-Toe CLI

A simple command-line Tic-Tac-Toe game in Python with board validation and winner detection.

## Features

- Play on a 3x3 board from the command line
- Validate moves and board state
- Detect winners (X or O)
- Supports making a move via command-line arguments

## Requirements

- Python 3.x

## Usage

Run the script:

```bash
python tictactoe.py
````

### Command-Line Arguments

* `-b`, `--board`
  The state of the board (9 characters: `.`, `X`, or `O`). Default: `.........`

* `-p`, `--player`
  The player making a move (`X` or `O`).

* `-c`, `--cell`
  The cell number to place the player's mark (1-9).

> Note: If you provide a player, you must also provide a cell.

### Examples

1. Show the empty board:

```bash
python tictactoe.py
```

2. Make a move as X in cell 5:

```bash
python tictactoe.py -p X -c 5
```

3. Start from a partially filled board:

```bash
python tictactoe.py -b "X.O...X.."
```

## Board Layout

```
-------------
| 1 | 2 | 3 |
-------------
| 4 | 5 | 6 |
-------------
| 7 | 8 | 9 |
-------------
```

`.` indicates an empty cell.
