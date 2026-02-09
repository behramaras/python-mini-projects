```md
# Tkinter Calculator

A simple desktop calculator application built with Python and Tkinter, supporting basic arithmetic operations through a graphical user interface.

## Features

- Basic arithmetic operations: addition, subtraction, multiplication, division
- Button-based and keyboard input support
- Real-time expression display
- Error handling for invalid expressions
- Clean and minimal GUI layout

## Technologies Used

Python 3, Tkinter (GUI library)

## How to Run

1. Make sure Python 3 is installed on your system
2. Save the file as `calculator.py`
3. Run the application using:

```bash
python calculator.py
````

## How to Use

* Click the number and operator buttons to build an expression
* Press **=** or hit **Enter** to evaluate
* Press **C** to clear the input
* You can also use your keyboard for input

## How It Works

* User input is stored in a stack inside the `Calculator` class
* Expressions are evaluated using Python’s `eval()` function
* The GUI is handled by the `Window` class using Tkinter widgets
* Events are bound to both mouse clicks and keyboard input

## Notes

* This project uses `eval()` for simplicity and is intended for learning purposes
* Not recommended for handling untrusted input in production environments
