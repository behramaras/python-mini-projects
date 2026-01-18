# Multiplication Table GUI

A simple desktop application built with **Python** and **PyQt5** to help users practice multiplication with different difficulty levels.

## Features

* Graphical User Interface (GUI) using PyQt5
* Random multiplication questions
* Multiple difficulty levels:

  * Easy
  * Medium
  * Hard
  * Very Hard
* Instant feedback with correct/incorrect answers
* Color-coded result display (green for correct, red for incorrect)

## Project Structure

```
multiplication-table-GUI/
├── main.py        # Application logic
├── form.ui        # Qt Designer UI file
└── README.md      # Project documentation
```

## Requirements

* Python 3.8+
* PyQt5

Install dependencies with:

```bash
pip install PyQt5
```

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/behramaras/python-mini-projects.git
   ```

2. Navigate to the project folder:

   ```bash
   cd python-mini-projects/multiplication-table-GUI
   ```

3. Run the application:

   ```bash
   python main.py
   ```

## How It Works

* Select a difficulty level from the dropdown
* Two random numbers are generated based on the selected level
* Enter your answer and press **Enter**
* The result is displayed immediately
* A new question is generated automatically

## UI Design

The interface is created using **Qt Designer** and loaded dynamically using:

```python
uic.loadUi("form.ui", self)
```

## Possible Improvements

* Add score tracking
* Add timer mode
* Save results to a file
* Sound or animation feedback

## License

This project is for educational purposes.
