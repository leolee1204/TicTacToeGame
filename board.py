class Board:
    EMPTY_CELL = 0
    PLAYER_X = 1
    PLAYER_O = -1

    def __init__(self):
        self.reset_board()

    def reset_board(self):
        self.game_board = [[self.EMPTY_CELL] * 3 for _ in range(3)]

    def print_board(self):
        print("\nPositions:")
        self.print_board_with_positions()

        print("Board:")
        for row in self.game_board:
            print("|", end="")
            for col in row:
                if col == self.EMPTY_CELL:
                    print("   |", end="")
                else:
                    print(f" {'X' if col == self.PLAYER_X else 'O'} |", end="")
            print()
        print()

    def print_board_with_positions(self):
        print("| 1 | 2 | 3 |\n| 4 | 5 | 6 |\n| 7 | 8 | 9 |")

    def step(self, move, player_marker):
        row, col = move.get_row(), move.get_column()

        if self.game_board[row][col] == self.EMPTY_CELL:
            self.game_board[row][col] = (
                self.PLAYER_X if player_marker == "X" else self.PLAYER_O
            )
            next_state = self.board_to_state()

            if self.check_winner(player_marker):
                return next_state, 1 if player_marker == "X" else -1, True, {}
            elif self.check_is_tie():
                return next_state, 0, True, {}
            else:
                return next_state, 0, False, {}
        else:
            return self.board_to_state(), -10, False, {}

    def check_winner(self, player_marker):
        player_value = self.PLAYER_X if player_marker == "X" else self.PLAYER_O

        # Check rows, columns, and diagonals
        for i in range(3):
            if all(self.game_board[i][j] == player_value for j in range(3)) or all(
                self.game_board[j][i] == player_value for j in range(3)
            ):
                return True

        if all(self.game_board[i][i] == player_value for i in range(3)) or all(
            self.game_board[i][2 - i] == player_value for i in range(3)
        ):
            return True

        return False

    def check_is_tie(self):
        return all(
            self.game_board[i][j] != self.EMPTY_CELL for i in range(3) for j in range(3)
        )

    def board_to_state(self):
        # flatten
        return [cell for row in self.game_board for cell in row]
