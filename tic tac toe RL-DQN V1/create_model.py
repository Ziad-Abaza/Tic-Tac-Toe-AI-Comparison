import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import random

#################################################
# Define the Environment
#################################################
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))  # Game board (3x3)
        self.current_player = 1  # Current player (1 or -1)
        self.done = False  # Is the game over?
        self.winner = 0  # Winner (1, -1, or 0 for a draw)

    def reset(self):
        # Reset the board and game state
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board.flatten()  # Return the board as a flattened list

    def step(self, action):
        # Execute a player's move
        row, col = divmod(action, 3)  # Convert action to row and column
        if self.board[row, col] != 0:  # If the move is invalid
            return self.board.flatten(), -10, True, {}  # Severe penalty for invalid move

        self.board[row, col] = self.current_player  # Place the player's mark
        winner = self.check_winner()  # Check if there's a winner
        if winner != 0:  # If there's a winner
            self.done = True
            self.winner = winner
            reward = 1 if winner == 1 else -1  # Reward based on the winner
            return self.board.flatten(), reward, self.done, {}

        if np.all(self.board != 0):  # If the board is full (draw)
            self.done = True
            return self.board.flatten(), 0, self.done, {}

        self.current_player *= -1  # Switch players
        return self.board.flatten(), 0, self.done, {}

    def check_winner(self):
        # Check rows, columns, and diagonals for a winner
        for i in range(3):
            if abs(sum(self.board[i])) == 3:  # Check rows
                return self.board[i][0]
            if abs(sum(self.board[:, i])) == 3:  # Check columns
                return self.board[0][i]
        if abs(self.board[0, 0] + self.board[1, 1] + self.board[2, 2]) == 3:  # Main diagonal
            return self.board[0, 0]
        if abs(self.board[0, 2] + self.board[1, 1] + self.board[2, 0]) == 3:  # Secondary diagonal
            return self.board[0, 2]
        return 0  # No winner

#################################################
# Define the Deep Q-Network (DQN) Model
#################################################
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # Size of the state (flattened board)
        self.action_size = action_size  # Number of possible actions (9 cells)
        self.memory = deque(maxlen=2000)  # Memory to store past experiences
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration vs exploitation rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.995  # Decay rate for exploration
        self.learning_rate = 0.001  # Learning rate for the model
        self.model = self._build_model()  # Build the neural network model

    def _build_model(self):
        # Define the neural network architecture
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),  # Input layer
            layers.Dense(24, activation='relu'),  # Hidden layer
            layers.Dense(self.action_size, activation='linear')  # Output layer
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Choose an action (exploration vs exploitation)
        if np.random.rand() <= self.epsilon:  # Exploration: choose a random action
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)  # Exploitation: choose the best action
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        # Train the model using past experiences
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:  # Decay exploration rate
            self.epsilon *= self.epsilon_decay

#################################################
# Main Script
#################################################
if __name__ == "__main__":
    env = TicTacToeEnv()  # Initialize the environment
    state_size = 9  # State size (3x3 board flattened)
    action_size = 9  # Number of possible actions (9 cells)
    agent = DQNAgent(state_size, action_size)  # Initialize the DQN agent
    episodes = 1000  # Number of training episodes
    batch_size = 32  # Batch size for training

    for e in range(episodes):
        state = env.reset()  # Reset the environment
        state = np.reshape(state, [1, state_size])  # Reshape the state
        for time in range(9):  # Maximum moves in Tic-Tac-Toe is 9
            action = agent.act(state)  # Choose an action
            next_state, reward, done, _ = env.step(action)  # Execute the action
            next_state = np.reshape(next_state, [1, state_size])  # Reshape the next state
            agent.remember(state, action, reward, next_state, done)  # Store the experience
            state = next_state  # Update the current state
            if done:
                print(f"Episode: {e+1}/{episodes}, Winner: {env.winner}, Epsilon: {agent.epsilon:.2f}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)  # Train the model

    # Test the model after training
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(9):
        action = agent.act(state)  # Choose the best action
        next_state, reward, done, _ = env.step(action)  # Execute the action
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        print(f"Board:\n{np.reshape(state, (3, 3))}")  # Display the board
        if done:
            print(f"Game Over! Winner: {env.winner}")
            break