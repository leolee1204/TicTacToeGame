import random

from board import Board
from move import Move


class Player:
    PLAYER_MARKER = "X"
    COMPUTER_MARKER = "O"

    def __init__(self, is_human=True, use_rl=False, q_table=None):
        self._is_human = is_human
        self._use_rl = use_rl
        self._marker = self.PLAYER_MARKER if is_human else self.COMPUTER_MARKER
        self.q_table = q_table

    @property
    def is_human(self):
        return self._is_human

    @property
    def marker(self):
        return self._marker

    def get_move(self, board):
        if self.is_human:
            return self.get_human_move()
        elif self._use_rl:
            return self.get_rl_move(board)
        else:
            return self.get_random_move(board)

    def get_human_move(self):
        while True:
            try:
                user_input = int(input("Please enter your move (1-9): "))
                move = Move(user_input)
                if move.is_valid():
                    return move
            except ValueError:
                pass
            print("Invalid input. Please enter a number between 1 and 9.")

    def get_random_move(self, board):
        available_positions = [
            i for i in range(9) if board.game_board[i // 3][i % 3] == Board.EMPTY_CELL
        ]
        return Move(random.choice(available_positions) + 1)

    def get_rl_move(self, board):
        state = tuple(board.board_to_state())
        available_actions = [i for i in range(9) if state[i] == Board.EMPTY_CELL]
        q_values = [self.q_table.get((state, a), 0.0) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        action = random.choice(best_actions)
        return Move(action + 1)
