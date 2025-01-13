import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from numba import njit

#################################################
# check winner: Check if there are three identical symbols in a row
#################################################
@njit
def check_winner(board):
    for i in range(3):
        # Check each row
        if abs(sum(board[i])) == 3:  
            return board[i][0]
        # Check each column
        if abs(sum(board[:, i])) == 3:
            return board[0][i]
    # Check main diagonal (from upper left corner to lower right corner)
    if abs(board[0, 0] + board[1, 1] + board[2, 2]) == 3: 
        return board[0, 0]
    # Check the reverse diagonal (from upper right corner to lower left corner)
    if abs(board[0, 2] + board[1, 1] + board[2, 0]) == 3:  
        return board[0, 2]
    return 0

#################################################
# minimax with alpha-beta pruning
#################################################
def minimax(board, depth, alpha, beta, maximizing_player):
    winner = check_winner(board)
    if winner != 0:
        return winner * (10 - depth)
    if np.all(board != 0):
        return 0

    if maximizing_player:
        max_eval = -np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = 1
                    eval = minimax(board, depth + 1, alpha, beta, False)
                    board[i, j] = 0
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
        return max_eval
    else:
        min_eval = np.inf
        for i in range(3):
            for j in range(3):
                if board[i, j] == 0:
                    board[i, j] = -1
                    eval = minimax(board, depth + 1, alpha, beta, True)
                    board[i, j] = 0
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
        return min_eval

#################################################
# find best move using minimax
#################################################
def find_best_move(board, player):
    # Check if there is an immediate winning move
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = player
                if check_winner(board) == player:
                    board[i, j] = 0
                    return (i, j)
                board[i, j] = 0

    # Check if the opponent has an immediate winning move and block it
    opponent = -player
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = opponent
                if check_winner(board) == opponent:
                    board[i, j] = 0
                    return (i, j)
                board[i, j] = 0

    # Control the center if available
    if board[1, 1] == 0:
        return (1, 1)

    # Use Minimax with Alpha-Beta Pruning to choose the best move
    best_move = None
    best_value = -np.inf if player == 1 else np.inf
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = player
                move_value = minimax(board, 0, -np.inf, np.inf, player == -1)
                board[i, j] = 0
                if (player == 1 and move_value > best_value) or (player == -1 and move_value < best_value):
                    best_value = move_value
                    best_move = (i, j)
    return best_move

#################################################
# create model DNN
#################################################
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(3, 3, 1)),  # Input shape (3 rows, 3 columns, 1 channel)
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),  # Add Dropout to avoid Overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),  # Add Dropout to avoid Overfitting
        layers.Dense(9, activation='softmax')  # 9 possible moves
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

#################################################
# generate data with improved strategy
#################################################
def generate_data(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        board = np.zeros((3, 3))
        for turn in range(9):
            player = 1 if turn % 2 == 0 else -1
            if np.random.rand() < 0.2:  # 10% chance to make a random move
                empty_positions = np.argwhere(board == 0)
                if len(empty_positions) == 0:
                    break
                move = empty_positions[np.random.choice(len(empty_positions))]
            else:
                move = find_best_move(board, player)
                if move is None:
                    empty_positions = np.argwhere(board == 0)
                    if len(empty_positions) == 0:
                        break
                    move = empty_positions[np.random.choice(len(empty_positions))]
            board[tuple(move)] = player
            X.append(board.flatten())
            y.append(move[0] * 3 + move[1])

            if check_winner(board) != 0:
                break

    return np.array(X), np.array(y)

#################################################
# Main script
#################################################
print("Generating data...")
X, y = generate_data(5000)
print("Data generated.")

# Reshape data to fit CNN layers
X = X.reshape(-1, 3, 3, 1)

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()

print("Starting training...")
model.fit(X_train, y_train, epochs=120, batch_size=32, validation_split=0.2)
print("Training finished.")

model.save('tic_tac_toe_dnn_model.h5')
print("Model saved.")

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Function to choose the best move
def get_best_move(board, model):
    flat_board = board.flatten().reshape(1, 3, 3, 1)
    predictions = model.predict(flat_board)
    sorted_indices = np.argsort(predictions[0])[::-1]  # Sort predictions from highest to lowest
    for move in sorted_indices:
        row, col = divmod(move, 3)
        if board[row, col] == 0:
            return row, col
    return None

# Test the move
board = np.zeros((3, 3))
model = tf.keras.models.load_model('tic_tac_toe_dnn_model.h5')
print(get_best_move(board, model))