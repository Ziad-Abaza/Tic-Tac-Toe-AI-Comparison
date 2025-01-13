from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# تحميل النموذج المدرب
model = tf.keras.models.load_model('tic_tac_toe_dnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # الحصول على بيانات اللوحة من الطلب
    data = request.get_json()
    flat_board = np.array(data['board'], dtype=np.float32)  # تحويل اللوحة إلى مصفوفة numpy
    board = flat_board.reshape(3, 3)  # تحويل اللوحة إلى مصفوفة 3x3
    board = np.expand_dims(board, axis=-1)  # إضافة قناة واحدة (تصبح الشكل 3x3x1)
    board = np.expand_dims(board, axis=0)  # إضافة بُعد الدُفعة (تصبح الشكل 1x3x3x1)

    # التنبؤ بالحركة التالية باستخدام النموذج
    q_values = model.predict(board, verbose=0)
    next_move = np.argmax(q_values)  # اختيار الحركة ذات القيمة الأعلى

    # التحقق من أن الحركة صالحة (الخلية فارغة)
    if flat_board[next_move] != 0:
        # إذا كانت الحركة غير صالحة، نبحث عن حركة صالحة
        sorted_indices = np.argsort(q_values[0])[::-1]  # ترتيب الحركات من الأفضل إلى الأسوأ
        for move in sorted_indices:
            if flat_board[move] == 0:
                next_move = move
                break

    print(f"Prediction: {q_values}, Next move: {next_move}")
    return jsonify({'next_move': int(next_move)})

if __name__ == '__main__':
    app.run(debug=True)