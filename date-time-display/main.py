# coding=utf-8

"""
    Timer usage
    A small application that displays the current date and time
"""

import time
import os
import platform


def clear_screen():
    if platform.system() == 'Linux' or platform.system() == 'Darwin':
        os.system('clear')
    else:
        os.system('cls')


while True:

    current_time = time.localtime()
    year = current_time[0]
    month = current_time[1]
    day = current_time[2]
    hour = current_time[3]
    minute = current_time[4]
    second = current_time[5]

    time.sleep(1)
    clear_screen()

    print(f"""
        date : {day}/{month}/{year}
        time : {hour}:{minute}:{second}
    """)
