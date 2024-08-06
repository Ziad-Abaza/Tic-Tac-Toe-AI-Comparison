from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app) 

model = tf.keras.models.load_model('tic_tac_toe_dnn_model_3.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    board = np.array(data['board']).reshape(1, 3, 3, 1)
    prediction = model.predict(board)
    next_move = np.argmax(prediction)
    
    # التحقق من أن الحركة صحيحة
    flat_board = data['board']
    if flat_board[next_move] != 0:
        # إذا كانت الحركة غير صالحة، حاول العثور على حركة أخرى صالحة
        sorted_indices = np.argsort(prediction[0])[::-1]
        for move in sorted_indices:
            if flat_board[move] == 0:
                next_move = move
                break
    
    print(f"Prediction: {prediction}, Next move: {next_move}")
    return jsonify({'next_move': int(next_move)})

if __name__ == '__main__':
    app.run(debug=True)
