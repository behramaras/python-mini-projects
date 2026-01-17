import sys
from random import randint

from PyQt5 import uic
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QComboBox,
    QLabel,
    QLineEdit
)

LEVELS = {
    "Easy": (1, 6),
    "Medium": (6, 12),
    "Hard": (12, 25),
    "Very Hard": (25, 100),
}


def get_random_numbers(level):
    start, end = LEVELS[level]
    num1 = randint(start, end)
    num2 = randint(start, end)
    return num1, num2


class MainWindow(QMainWindow):
    cb_level: QComboBox
    lbl_num1: QLabel
    lbl_num2: QLabel
    lbl_result: QLabel
    txt_result: QLineEdit
    statusbar: QStatusBar

    num1: int = 0
    num2: int = 0
    current_level = "Easy"

    def __init__(self, flags=None, *args, **kwargs):
        super().__init__(flags, *args, **kwargs)

        uic.loadUi("form.ui", self)
        self.initialize()

    def initialize(self):
        self.cb_level.currentTextChanged.connect(
            lambda text: self.on_level_changed(text)
        )
        self.txt_result.returnPressed.connect(self.on_result_entered)
        self.txt_result.setFocus()
        self.start()

    def on_level_changed(self, text):
        self.current_level = text
        self.start()

    def start(self):
        self.num1, self.num2 = get_random_numbers(self.current_level)
        self.lbl_num1.setText(str(self.num1))
        self.lbl_num2.setText(str(self.num2))

    def on_result_entered(self):
        correct_answer = str(self.num1 * self.num2)
        user_answer = self.txt_result.text()
        self.txt_result.setText("")

        if correct_answer == user_answer:
            result_text = f"{self.num1} x {self.num2} = {correct_answer}"
            self.lbl_result.setStyleSheet("color: green")
        else:
            result_text = f"{self.num1} x {self.num2} = {user_answer} → {correct_answer}"
            self.lbl_result.setStyleSheet("color: red")

        self.lbl_result.setText(result_text)
        self.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
