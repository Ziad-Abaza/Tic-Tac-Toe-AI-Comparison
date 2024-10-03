# Tic-Tac-Toe AI

This project implements an AI for the classic Tic-Tac-Toe game using a Deep Neural Network (DNN) built with TensorFlow and a Flask web application for the frontend.

## Features

- Play against an AI that learns optimal moves.
- User-friendly web interface.
- Predicts the next best move based on the current board state.
- Uses a DNN model to evaluate game states.

## Requirements

- Python 3.x
- TensorFlow
- Flask
- Flask-CORS
- NumPy
- scikit-learn

You can install the required packages using:

```bash
pip install tensorflow flask flask-cors numpy scikit-learn
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Ziad-Abaza/tic_tac_toe-AI.git
   cd tic-tac-toe-ai
   ```

2. Ensure you have all the required packages installed as mentioned in the Requirements section.

## How to Run

1. Train the model (if not already trained):
   - Run the `create_model.py` script to generate training data and train the model:
   ```bash
   python create_model.py
   ```

2. Start the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and go to `http://127.0.0.1:5000` to access the game.

## Usage

- Click on the squares in the Tic-Tac-Toe grid to make your move.
- The AI will respond with its move based on the current state of the board.

## Model Training

The model is trained on random game states where it learns to predict the best move based on the current configuration of the board. The training data is generated using the `generate_data` function in `create_model.py`, where multiple game states are simulated.

### Model Structure

- The model consists of:
  - Input layer flattening the 3x3 board.
  - Two hidden dense layers with ReLU activation.
  - Output layer with softmax activation to represent the probabilities of the next moves.

## File Structure

```
tic-tac-toe-ai/
│
├── create_model.py          # Script to create and train the DNN model.
├── app.py                   # Flask application for the web interface.
├── templates/
│   └── index.html           # HTML file for the web interface.
└── static/
    ├── style.css            # CSS for styling the web application.
    └── script.js            # JavaScript for game functionality.
```
