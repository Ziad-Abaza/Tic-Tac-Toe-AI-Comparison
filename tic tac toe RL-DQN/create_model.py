import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import random

#################################################
# تعريف البيئة (Environment)
#################################################
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))  # لوحة اللعبة
        self.current_player = 1  # اللاعب الحالي (1 أو -1)
        self.done = False  # هل انتهت اللعبة؟
        self.winner = 0  # الفائز (1 أو -1 أو 0 للتعادل)

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        self.done = False
        self.winner = 0
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:  # الحركة غير صالحة
            return self.board.flatten(), -10, True, {}  # عقاب شديد للحركة غير الصالحة

        self.board[row, col] = self.current_player
        winner = self.check_winner()
        if winner != 0:  # فوز
            self.done = True
            self.winner = winner
            reward = 1 if winner == 1 else -1
            return self.board.flatten(), reward, self.done, {}

        if np.all(self.board != 0):  # تعادل
            self.done = True
            return self.board.flatten(), 0, self.done, {}

        self.current_player *= -1  # تبديل اللاعب
        return self.board.flatten(), 0, self.done, {}

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i])) == 3:  # الصفوف
                return self.board[i][0]
            if abs(sum(self.board[:, i])) == 3:  # الأعمدة
                return self.board[0][i]
        if abs(self.board[0, 0] + self.board[1, 1] + self.board[2, 2]) == 3:  # القطر الرئيسي
            return self.board[0, 0]
        if abs(self.board[0, 2] + self.board[1, 1] + self.board[2, 0]) == 3:  # القطر الثانوي
            return self.board[0, 2]
        return 0

#################################################
# تعريف نموذج Deep Q-Network (DQN)
#################################################
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # ذاكرة للتجارب السابقة
        self.gamma = 0.95  # عامل الخصم
        self.epsilon = 1.0  # استكشاف vs استغلال
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(24, input_dim=self.state_size, activation='relu'),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:  # استكشاف
            return random.randrange(self.action_size)
        q_values = self.model.predict(state, verbose=0)  # استغلال
        return np.argmax(q_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state, verbose=0)[0])
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

#################################################
# Main script
#################################################
if __name__ == "__main__":
    env = TicTacToeEnv()
    state_size = 9  # حجم الحالة (3x3 لوحة)
    action_size = 9  # عدد الحركات الممكنة (9 خلايا)
    agent = DQNAgent(state_size, action_size)
    episodes = 1000  # عدد الحلقات التدريبية
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(9):  # الحد الأقصى للحركات في لعبة Tic-Tac-Toe هو 9
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e+1}/{episodes}, Winner: {env.winner}, Epsilon: {agent.epsilon:.2f}")
                break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

    # اختبار النموذج بعد التدريب
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(9):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        print(f"Board:\n{np.reshape(state, (3, 3))}")
        if done:
            print(f"Game Over! Winner: {env.winner}")
            break