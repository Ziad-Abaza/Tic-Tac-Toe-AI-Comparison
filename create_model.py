import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# دالة للتحقق من الفائز
def check_winner(board):
    for i in range(3):
        if abs(sum(board[i])) == 3:  # تحقق من الصفوف
            return board[i][0]
        if abs(sum(board[:, i])) == 3:  # تحقق من الأعمدة
            return board[0][i]
    if abs(board[0, 0] + board[1, 1] + board[2, 2]) == 3:  # تحقق من القطر الرئيسي
        return board[0, 0]
    if abs(board[0, 2] + board[1, 1] + board[2, 0]) == 3:  # تحقق من القطر العكسي
        return board[0, 2]
    return 0

# دالة للبحث عن الحركة لمنع الخصم من الفوز أو للفوز
def find_best_move(board, player):
    # التحقق من الفرص لمنع الخصم من الفوز
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = -player
                if check_winner(board) == -player:
                    board[i, j] = 0
                    return i, j
                board[i, j] = 0

    # التحقق من الفرص للفوز
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = player
                if check_winner(board) == player:
                    board[i, j] = 0
                    return i, j
                board[i, j] = 0

    # اختيار أي حركة متاحة أخرى
    empty_positions = np.argwhere(board == 0)
    if len(empty_positions) > 0:
        return empty_positions[np.random.choice(len(empty_positions))]

    return None

# بديل Minimax: شبكة عصبية لتوقع الحركات
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(3, 3, 1)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(9, activation='softmax')  # 9 خيارات للحركات
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_data(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        board = np.zeros((3, 3))
        for turn in range(9):
            player = 1 if turn % 2 == 0 else -1

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

print("Generating data...")
X, y = generate_data(50000)
print("Data generated.")

# تحويل البيانات لتناسب طبقات CNN
X = X.reshape(-1, 3, 3, 1)

# تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()

print("Starting training...")
model.fit(X_train, y_train, epochs=120, batch_size=64, validation_split=0.2)
print("Training finished.")

model.save('tic_tac_toe_dnn_model_4.h5')
print("Model saved.")

# تقييم النموذج
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# دالة لاختيار أفضل حركة
def get_best_move(board, model):
    flat_board = board.flatten().reshape(1, 3, 3, 1)
    predictions = model.predict(flat_board)
    sorted_indices = np.argsort(predictions[0])[::-1]  # ترتيب التوقعات من الأكبر للأصغر
    for move in sorted_indices:
        row, col = divmod(move, 3)
        if board[row, col] == 0:
            return row, col
    return None

# اختبار الحركة
board = np.zeros((3, 3))
model = tf.keras.models.load_model('tic_tac_toe_dnn_model_4.h5')
print(get_best_move(board, model))
