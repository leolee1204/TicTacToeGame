from flask import Flask, render_template, request, jsonify
import os
import sys
from board import Board
from player import Player
from train import QLearningAgent

app = Flask(__name__)


def resource_path(relative_path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        return os.path.join(os.path.abspath("."), relative_path)


class TicTacToeGame:
    def __init__(self):
        self.board = Board()
        self.agent = QLearningAgent()
        self.agent.load_model(resource_path("q_table.pkl"))
        self.human_player = Player(is_human=True)
        self.ai_player = Player(is_human=False, use_rl=True, q_table=self.agent.q_table)
        self.current_player = self.human_player

    def make_move(self, move):
        state, reward, done, _ = self.board.step(move, self.current_player.marker)
        self.current_player = (
            self.ai_player
            if self.current_player == self.human_player
            else self.human_player
        )
        return self.get_board_state(), done

    def ai_move(self):
        state = self.board.board_to_state()
        available_actions = [
            i for i, cell in enumerate(state) if cell == self.board.EMPTY_CELL
        ]
        action = self.agent.choose_action(state, available_actions)
        return self.make_move(action + 1)

    def check_game_end(self):
        if self.board.check_winner("X"):
            return "You win!"
        elif self.board.check_winner("O"):
            return "AI wins!"
        elif self.board.check_is_tie():
            return "It's a tie!"
        return None

    def reset_game(self):
        self.board.reset_board()
        self.current_player = self.human_player

    def get_board_state(self):
        return [
            (
                "X"
                if cell == self.board.PLAYER_X
                else "O" if cell == self.board.PLAYER_O else ""
            )
            for cell in self.board.board_to_state()
        ]


game = TicTacToeGame()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/make_move", methods=["POST"])
def make_move():
    move = int(request.json["move"])
    state, _ = game.make_move(move)
    game_over = game.check_game_end()

    if not game_over:
        ai_state, _ = game.ai_move()
        game_over = game.check_game_end()
    else:
        ai_state = state

    return jsonify(
        {
            "board": ai_state,
            "game_over": game_over,
            "current_player": "X" if game.current_player == game.human_player else "O",
        }
    )


@app.route("/reset_game", methods=["POST"])
def reset_game():
    game.reset_game()
    return jsonify(
        {"success": True, "board": game.get_board_state(), "current_player": "X"}
    )


if __name__ == "__main__":
    if not os.path.exists(resource_path("q_table.pkl")):
        # Train the AI if the model doesn't exist
        from game import TicTacToeGame as TrainingGame

        training_game = TrainingGame()
        training_game.train_ai()
    app.run(debug=True)
