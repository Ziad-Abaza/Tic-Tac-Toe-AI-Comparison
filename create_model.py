import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
#################################################
# check winner: للتحقق من وجود ثلاث رموز متطابقة على التوالي
#################################################
def check_winner(board):
    for i in range(3):
        #Check each row
        if abs(sum(board[i])) == 3:  
            return board[i][0]
        # Check each column
        if abs(sum(board[:, i])) == 3:
            return board[0][i]
    # Check main diameter (from upper left corner to lower right corner)
    if abs(board[0, 0] + board[1, 1] + board[2, 2]) == 3: 
        return board[0, 0]
    # Check the reverse diameter (from the upper right corner to the lower left corner)
    if abs(board[0, 2] + board[1, 1] + board[2, 0]) == 3:  
        return board[0, 2]
    return 0
#################################################
# find best move:  البحث عن الحركة لمنع الخصم من الفوز أو للفوز
#################################################
def find_best_move(board, player):
    # التحقق من الفرص لمنع الخصم من الفوز
    # عن طريق وضع رمز الخصم في هذه الخلية وتتحقق إذا كان الخصم سيفوز بهذه الحركة
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = -player
                if check_winner(board) == -player:
                    board[i, j] = 0
                    return i, j
                board[i, j] = 0

    # التحقق من الفرص للفوز
    # عن طريق وضع رمز اللاعب في هذه الخلية وتتحقق إذا كانت هذه الحركة ستؤدي إلى فوز اللاعب.
    # إذا كانت الحركة ستؤدي إلى فوز اللاعب، تعيد الدالة موقع هذه الخلية لتحقيق الفوز.
    for i in range(3):
        for j in range(3):
            if board[i, j] == 0:
                board[i, j] = player
                if check_winner(board) == player:
                    board[i, j] = 0
                    return i, j
                board[i, j] = 0

    # تختار حركة عشوائية من بين الخلايا الفارغة المتاحة
    empty_positions = np.argwhere(board == 0)
    if len(empty_positions) > 0:
        return empty_positions[np.random.choice(len(empty_positions))]

    return None
#################################################
# create model DNN
#################################################
def create_model():
    model = models.Sequential([
        layers.Flatten(input_shape=(3, 3, 1)), # شكل الإدخال الذي يمثل لوح اللعبة (3 صفوف، 3 أعمدة، و1 قناة للون)
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(9, activation='softmax')  # 9 خيارات للحركات
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_data(num_samples):
    # X: قائمة لتخزين اللوحات المسطحة.
    # y: قائمة لتخزين الحركات المثلى.
    X = []
    y = []
    for _ in range(num_samples):
        # إنشاء لوحة فارغة بحجم 3*3
        board = np.zeros((3, 3))
        # تكرار بعدد الحركات في اللعبة
        for turn in range(9):
        # تحديد اللاعب (1:X,-1:O) بالتناوب
            player = 1 if turn % 2 == 0 else -1
            # البحث عن الحركة المثلى
            move = find_best_move(board, player)
            # لم يتم العثور على حركة مثلى
            if move is None:
                empty_positions = np.argwhere(board == 0) #  العثور على الأماكن الفارغة في اللوحة
                if len(empty_positions) == 0:
                    break
                move = empty_positions[np.random.choice(len(empty_positions))] #  اختيار حركة عشوائية من الأماكن الفارغة المتبقية
            # تحديث اللوحة بالحركة المثلى
            board[tuple(move)] = player
            X.append(board.flatten())
            y.append(move[0] * 3 + move[1])

            if check_winner(board) != 0:
                break

    return np.array(X), np.array(y)

print("Generating data...")
X, y = generate_data(10000)
print("Data generated.")

# تحويل البيانات لتناسب طبقات CNN
X = X.reshape(-1, 3, 3, 1)

# تقسيم البيانات للتدريب والاختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = create_model()

print("Starting training...")
model.fit(X_train, y_train, epochs=70, batch_size=32, validation_split=0.2)
print("Training finished.")

model.save('tic_tac_toe_dnn_model_3.h5')
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
model = tf.keras.models.load_model('tic_tac_toe_dnn_model_3.h5')
print(get_best_move(board, model))
