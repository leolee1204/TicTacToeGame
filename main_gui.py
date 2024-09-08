import os
import sys

from board import Board
from game import TicTacToeGame
from player import Player
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QHBoxLayout, QLabel, QMessageBox,
                             QPushButton, QVBoxLayout, QWidget)
from train import QLearningAgent


def resource_path(relative_path):
    """Get the absolute path to the resource (works for both dev and PyInstaller bundled mode)"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.abspath("."), relative_path)


class TicTacToeGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.board = Board()
        self.agent = QLearningAgent()
        self.agent.load_model(resource_path("q_table.pkl"))  # Load the trained model
        self.human_player = Player(is_human=True)
        self.ai_player = Player(is_human=False, use_rl=True, q_table=self.agent.q_table)
        self.current_player = self.human_player
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Design Tic-Tac-Toe-Game By Leo")
        self.setGeometry(300, 300, 300, 350)

        layout = QVBoxLayout()

        self.status_label = QLabel("Your turn (X)")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color: black;color: white;")
        layout.addWidget(self.status_label)

        self.buttons = []
        for i in range(3):
            row_layout = QHBoxLayout()
            for j in range(3):
                button = QPushButton("")
                button.setFixedSize(80, 80)
                button.setFont(QFont("Arial", 40, QFont.Bold))
                button.setStyleSheet("background-color: black; color: white;")
                button.clicked.connect(
                    lambda _, row=i, col=j: self.on_button_click(row, col)
                )
                row_layout.addWidget(button)
                self.buttons.append(button)
            layout.addLayout(row_layout)

        reset_button = QPushButton("Reset Game")
        reset_button.setFixedSize(300, 60)
        reset_button.setFont(QFont("Arial", 20, QFont.Bold))
        reset_button.setStyleSheet("background-color: black; color: white;")
        reset_button.clicked.connect(self.reset_game)
        layout.addWidget(reset_button)

        quit_button = QPushButton("Quit Game")
        quit_button.setFixedSize(300, 60)
        quit_button.setFont(QFont("Arial", 20, QFont.Bold))
        quit_button.setStyleSheet("background-color: black; color: white;")
        quit_button.clicked.connect(QApplication.quit)
        layout.addWidget(quit_button)

        self.setLayout(layout)

    def on_button_click(self, row, col):
        if (
            self.current_player.is_human
            and self.board.game_board[row][col] == self.board.EMPTY_CELL
        ):
            move = row * 3 + col + 1
            self.make_move(move)
            if not self.check_game_end():
                self.ai_move()

    def make_move(self, move):
        state, reward, done, _ = self.board.step(move, self.current_player.marker)
        button_index = move - 1
        self.set_button_text(self.buttons[button_index], self.current_player.marker)
        self.current_player = (
            self.ai_player
            if self.current_player == self.human_player
            else self.human_player
        )
        self.status_label.setText(
            f"{'Your' if self.current_player.is_human else 'AI'} turn ({'X' if self.current_player.marker == 'X' else 'O'})"
        )

    def set_button_text(self, button, text):
        button.setText(text)
        if text == "X":
            button.setStyleSheet("background-color: black;color: green;")
        elif text == "O":
            button.setStyleSheet("background-color: black;color: red;")

    def ai_move(self):
        state = self.board.board_to_state()
        available_actions = [i for i in range(9) if state[i] == self.board.EMPTY_CELL]
        action = self.agent.choose_action(state, available_actions)
        self.make_move(action + 1)
        self.check_game_end()

    def check_game_end(self):
        if self.board.check_winner("X"):
            self.game_over("You win!")
            return True
        elif self.board.check_winner("O"):
            self.game_over("AI wins!")
            return True
        elif self.board.check_is_tie():
            self.game_over("It's a tie!")
            return True
        return False

    def game_over(self, message):
        QMessageBox.information(self, "Game Over", message)
        self.reset_game()

    def reset_game(self):
        self.board.reset_board()
        self.current_player = self.human_player
        self.status_label.setText("Your turn (X)")
        for button in self.buttons:
            button.setText("")


if __name__ == "__main__":
    if not os.path.exists(resource_path("q_table.pkl")):
        game = TicTacToeGame()
        game.train_ai()
    app = QApplication(sys.argv)
    ex = TicTacToeGUI()
    ex.show()
    sys.exit(app.exec_())
