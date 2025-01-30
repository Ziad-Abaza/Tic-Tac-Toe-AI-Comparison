import numpy as np
import tensorflow as tf
from collections import deque
import random

class TicTacToeEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        return self.get_state()
    
    def get_state(self):
        return np.reshape(self.board, (3, 3, 1))
    
    def get_valid_moves(self):
        return [i for i, val in enumerate(self.board) if val == 0]
    
    def step(self, action):
        if self.board[action] != 0 or self.game_over:
            return self.get_state(), -10, True  
        
        self.board[action] = self.current_player
        
        if self.check_winner(self.current_player):
            reward = 10  # Increased reward for winning
            self.game_over = True
            self.winner = self.current_player
        elif 0 not in self.board:
            reward = 1  # Smaller reward for a draw
            self.game_over = True
        else:
            reward = -0.1  # Small penalty for not winning or drawing
            self.current_player = -1 if self.current_player == 1 else 1
        
        return self.get_state(), reward, self.game_over
    
    def check_winner(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in win_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False

class DQNAgent:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 32
        self.update_target_every = 10
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(3, 3, 1)),  
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(9, activation='linear') 
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_moves)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        act_values = act_values[0]
        valid_actions = np.argsort(act_values)[::-1]
        for action in valid_actions:
            if action in valid_moves:
                return action
        return np.random.choice(valid_moves)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        targets = self.model.predict(states)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                next_action = np.argmax(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                next_q = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0][next_action]
                target = reward + self.gamma * next_q
            
            targets[i][action] = target
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=1000):
        env = TicTacToeEnvironment()
        self.update_target_model()
        total_rewards = []
        wins = 0
        losses = 0
        draws = 0
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                valid_moves = env.get_valid_moves()
                action = self.act(state, valid_moves)
                next_state, reward, done = env.step(action)
                
                if reward == -10:
                    continue
                
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                
                if not done:
                    valid_moves = env.get_valid_moves()
                    opp_action = self.act(state, valid_moves)
                    next_state, _, done = env.step(opp_action)
                    state = next_state
                
                self.replay()
            
            total_rewards.append(total_reward)
            
            if env.winner == 1:
                wins += 1
            elif env.winner == -1:
                losses += 1
            else:
                draws += 1
            
            if episode % self.update_target_every == 0:
                self.update_target_model()
            
            if (episode+1) % 100 == 0:
                avg_reward = np.mean(total_rewards[-100:])
                print(f"Episode: {episode+1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")
                print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
        
        self.model.save('tic_tac_toe.h5')

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train(episodes=1000)