# Number Guessing Game

A simple Python console game where the player tries to guess a randomly generated number within a limited number of attempts.

## Description

In this game, the computer randomly generates a number between **1** and **100**. The player has **5 attempts** to guess the correct number. After each guess, the game provides a hint whether the guess should be higher or lower. The game ends when the player guesses the number correctly or runs out of attempts.

## How to Play

1. Run the Python script:

   ```bash
   python guessing_game.py
   ```
2. Enter your guess when prompted.
3. The game will tell you if your guess is too high or too low.
4. Continue guessing until:

   * You guess the correct number (you win), or
   * You run out of guesses (game over).

## Features

* Random number generation between 1 and 100.
* Limited number of guesses (5 by default).
* Hints provided for higher or lower guesses.
* Clear messages when you win or lose.

## Example

```
Please enter your guess: 50
Wrong guess, try a bigger number.
Remaining guesses: 4
Please enter your guess: 75
Wrong guess, try a smaller number.
Remaining guesses: 3
Please enter your guess: 63
Congratulations! You won.
```
