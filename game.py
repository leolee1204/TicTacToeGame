import os

from board import Board
from player import Player
from train import QLearningAgent, train_with_self_play


class TicTacToeGame:
    def __init__(self):
        self.agent = QLearningAgent()
        self.q_table_file = "q_table.pkl"

    def start(self):
        print("*" * 20)
        print("     Welcome to Tic-Tac-Toe     ")
        print("*" * 20)

        # Load or create Q-table
        if os.path.exists(self.q_table_file):
            self.agent.load_model(self.q_table_file)
        else:
            print("No existing model found. Starting with a new model.")

        while True:
            choice = input(
                "Do you want to (1) Play a game or (2) Train the AI? Enter 1 or 2: "
            ).strip()
            if choice == "1":
                self.start_new_game()
            elif choice == "2":
                self.train_ai()
            else:
                print("Invalid choice. Please enter 1 or 2.")
                continue

            play_again = (
                input("Would you like to continue? [yes|y]/[no|n]: ").strip().lower()
            )
            if play_again in ["no", "n"]:
                print("Bye! Come back soon")
                break
            elif play_again not in ["yes", "y"]:
                print("Invalid input, but I'll assume you want to continue!")

    def start_new_game(self):
        board = Board()
        human_player = Player(is_human=True)
        computer_player = Player(
            is_human=False, use_rl=True, q_table=self.agent.q_table
        )
        current_player = human_player

        game_history = []
        state = board.board_to_state()

        while True:
            board.print_board()
            move = current_player.get_move(board)
            next_state, reward, done, _ = board.step(move, current_player.marker)

            if not current_player.is_human:
                game_history.append((state, move.value - 1))

            if done:
                board.print_board()
                if reward == 0:
                    print("It's a tie! Try again.")
                    final_reward = 0
                else:
                    winner = "You" if current_player.is_human else "Computer"
                    print(f"Awesome, {winner} win!")
                    final_reward = reward if not current_player.is_human else -reward

                # Update Q-table
                for state, action in reversed(game_history):
                    available_actions = [
                        i
                        for i in range(9)
                        if board.game_board[i // 3][i % 3] == Board.EMPTY_CELL
                    ]
                    self.agent.update_q_table(
                        state, action, final_reward, next_state, available_actions
                    )
                    final_reward *= (
                        self.agent.gamma
                    )  # Discount the reward for earlier actions

                break

            state = next_state
            current_player = (
                human_player if current_player == computer_player else computer_player
            )

        # Save the updated Q-table after each game
        self.agent.save_model(self.q_table_file)

    def train_ai(self):
        num_games = int(input("How many training games do you want to play? "))
        train_with_self_play(self.agent, num_games)
        # train_with_human_play(self.agent, num_games)
        self.agent.save_model(self.q_table_file)
