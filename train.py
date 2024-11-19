import pickle
import random

from tqdm import tqdm

from board import Board
from move import Move


class QLearningAgent:
    """
    alpha（學習率）
        較高alpha（接近 1）意味著智能體更重視新訊息，學習速度很快，但可能會變得不穩定。
        較低alpha（接近 0）意味著代理學習緩慢，嚴重依賴先驗知識，但可能會錯過重要的更新。

    gamma（折扣係數）
        越高gamma（接近 1）代表代理商更重視未來的獎勵，從而製定更長期的計畫。
        較低gamma（接近 0）意味著代理人更專注於即時獎勵，行為更加短視。

    epsilon（探索率）
        較高epsilon（接近 1）意味著智能體探索更多，嘗試新的行動。
        較低epsilon（接近 0）意味著代理利用更多，依賴當前的 Q 值並選擇最著名的操作。
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def choose_action(self, state, available_actions):
        # epsilon 探索率
        if random.random() < self.epsilon:
            return random.choice(available_actions)

        q_values = [self.get_q_value(state, a) for a in available_actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update_q_table(self, state, action, reward, next_state, available_actions):
        old_q = self.get_q_value(state, action)
        next_max_q = max(
            [self.get_q_value(next_state, a) for a in available_actions], default=0
        )
        """
        old_qstate：這是狀態-動作對 ( , )的當前 Q 值action。它代表智能體對在該狀態下採取該操作的預期未來獎勵的當前估計。
        reward：在當前狀態下採取行動後立即獲得的獎勵。
        next_max_q：這是下一個狀態 ( ) 的最大 Q 值next_state。它代表智能體對從該點開始的最佳可能未來獎勵的估計（透過查看下一個狀態中的最佳可能行動來計算）。
        
        new_q = 舊的q值 + 學習率 * (當下獎勵 + 折扣因子*下一步最大q值 - 舊的q值)
        """
        new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
        self.q_table[(tuple(state), action)] = new_q

    def save_model(self, filename="q_learning_model.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_model(self, filename="q_learning_model.pkl"):
        try:
            with open(filename, "rb") as f:
                self.q_table = pickle.load(f)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No existing model found. Starting with a new model.")


def play_self_training_game(agent, board):
    """
    slef play with train
    """
    state = board.board_to_state()
    game_history = []
    current_player = Board.PLAYER_X

    while True:
        available_actions = [i for i in range(9) if state[i] == Board.EMPTY_CELL]
        action = agent.choose_action(state, available_actions)
        move = Move(action + 1)
        game_history.append((state, action, current_player))
        """
        board.step(move, player_marker)
        next_state, reward, done, _ = next_state, 0, False, {}
        """
        next_state, reward, done, _ = board.step(
            move, "X" if current_player == Board.PLAYER_X else "O"
        )

        if done:
            # Adjust reward based on the current player's perspective
            adjusted_reward = reward if current_player == Board.PLAYER_X else -reward
            return game_history, adjusted_reward

        state = next_state
        # swithch player
        current_player = (
            Board.PLAYER_O if current_player == Board.PLAYER_X else Board.PLAYER_X
        )


def train_with_self_play(agent, num_games=10000):
    """
    training 20 minllions times
    """
    board = Board()

    # tqdm 進度條
    rewards = []
    for _ in tqdm(range(num_games), desc="Training Progress"):
        board.reset_board()
        # agent:QLearningAgent()
        game_history, final_reward = play_self_training_game(agent, board)
        rewards.append(final_reward)

        # Update Q-table
        for state, action, player in reversed(game_history):
            # next_state :[1,0,-1,0,0,1,-1,0,0]]
            next_state = board.board_to_state()
            # available_actions [1,3,4,7,8]
            available_actions = [
                i for i in range(9) if next_state[i] == Board.EMPTY_CELL
            ]

            # Adjust reward based on the player's perspective
            player_reward = final_reward if player == Board.PLAYER_X else -final_reward

            agent.update_q_table(
                state, action, player_reward, next_state, available_actions
            )
            final_reward *= agent.gamma  # Discount the reward for earlier actions

    agent.save_model()
    average_reward = sum(rewards) / len(rewards)
    print(average_reward)
