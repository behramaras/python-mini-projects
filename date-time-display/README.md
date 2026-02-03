````markdown
# Terminal Timer

A small Python application that displays the current date and time in the terminal and updates every second.

## Features

- Shows current date (day/month/year)
- Shows current time (hour:minute:second)
- Automatically refreshes every second
- Clears the terminal screen for a clean display
- Works on Windows, Linux, and macOS

## Requirements

- Python 3.x

No external libraries are required.

## Usage

1. Clone or download the repository.
2. Open a terminal in the project directory.
3. Run the script:

```bash
python timer.py
````

The terminal will continuously display the current date and time, updating every second.

## How It Works

* Uses `time.localtime()` to get the current date and time
* Uses `time.sleep(1)` to update every second
* Clears the terminal screen using OS-specific commands

## Notes

* Stop the program with `Ctrl + C`.

````
