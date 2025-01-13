from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('tic_tac_toe_dnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    flat_board = np.array(data['board'], dtype=np.float32)
    board = flat_board.reshape(3, 3)
    board = np.expand_dims(board, axis=-1)
    board = np.expand_dims(board, axis=0)

    q_values = model.predict(board, verbose=0)
    next_move = np.argmax(q_values)

    if flat_board[next_move] != 0:
        sorted_indices = np.argsort(q_values[0])[::-1]
        for move in sorted_indices:
            if flat_board[move] == 0:
                next_move = move
                break

    print(f"Prediction: {q_values}, Next move: {next_move}")
    return jsonify({'next_move': int(next_move)})

if __name__ == '__main__':
    app.run(debug=True)