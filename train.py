import pickle
import random

from board import Board
from move import Move
from player import Player
from tqdm import tqdm


class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
    
    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)
    
    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        
        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update_q_table(self, state, action, reward, next_state, available_actions):
        old_q = self.get_q_value(state, action)
        next_max_q = max([self.get_q_value(next_state, a) for a in available_actions], default=0)
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(tuple(state), action)] = new_q

    def save_model(self, filename='q_learning_model.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename='q_learning_model.pkl'):
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No existing model found. Starting with a new model.")

def play_training_game(agent, board):
    state = board.board_to_state()
    game_history = []

    while True:
        # Human player's turn
        board.print_board()
        while True:
            try:
                human_move = int(input("Enter your move (1-9): "))
                if human_move < 1 or human_move > 9:
                    raise ValueError
                move = Move(human_move)
                next_state, reward, done, _ = board.step(move, Player.PLAYER_MARKER)
                break
            except ValueError:
                print("Invalid input. Please enter a number between 1 and 9.")
            except Exception as e:
                print(f"Invalid move: {e}")

        if done:
            board.print_board()
            if reward == -1:
                print("You win!")
            else:
                print("It's a draw!")
            return game_history, -reward  # Return negative reward as the agent lost

        # AI agent's turn
        available_actions = [i for i in range(9) if board.game_board[i // 3][i % 3] == Board.EMPTY_CELL]
        action = agent.choose_action(next_state, available_actions)
        move = Move(action + 1)
        game_history.append((state, action))
        state, reward, done, _ = board.step(move, Player.COMPUTER_MARKER)

        if done:
            board.print_board()
            if reward == 1:
                print("AI wins!")
            else:
                print("It's a draw!")
            return game_history, reward

def train_with_human_play(agent, num_games=10):
    board = Board()
    
    for game in range(num_games):
        print(f"\nGame {game + 1}")
        board.reset_board()
        game_history, final_reward = play_training_game(agent, board)

        # Update Q-table
        for state, action in reversed(game_history):
            next_state = board.board_to_state()
            available_actions = [i for i in range(9) if board.game_board[i // 3][i % 3] == Board.EMPTY_CELL]
            agent.update_q_table(state, action, final_reward, next_state, available_actions)
            final_reward *= agent.gamma  # Discount the reward for earlier actions

        print(f"Game {game + 1} completed. AI final reward: {final_reward}")

    agent.save_model()
    print("Training completed and model saved.")

