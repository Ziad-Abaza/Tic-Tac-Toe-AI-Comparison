let title = document.querySelector(".title");
let statusText = document.getElementById("status");
let squares = [];
let turn = "X";
let gameOver = false;
let intervalId;

function TheEnd(num1, num2, num3) {
  statusText.innerHTML = `${squares[num1]} is the winner!`;
  statusText.classList.add("winner");
  document.getElementById("item" + num1).style.backgroundColor = "#4caf50";
  document.getElementById("item" + num2).style.backgroundColor = "#4caf50";
  document.getElementById("item" + num3).style.backgroundColor = "#4caf50";

  clearInterval(intervalId);
  intervalId = setInterval(function () {
    statusText.innerHTML += ".";
  }, 1000);

  setTimeout(function () {
    location.reload();
  }, 4000);
}

function winner() {
  let draw = true;

  for (let i = 1; i < 10; i++) {
    squares[i] = document.getElementById("item" + i).innerHTML;
    if (squares[i] === "") {
      draw = false;
    }
  }

  const winningCombinations = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [1, 4, 7],
    [2, 5, 8],
    [3, 6, 9],
    [1, 5, 9],
    [3, 5, 7],
  ];

  for (const combo of winningCombinations) {
    const [a, b, c] = combo;
    if (
      squares[a] === squares[b] &&
      squares[b] === squares[c] &&
      squares[a] !== ""
    ) {
      gameOver = true;
      TheEnd(a, b, c);
      return;
    }
  }

  if (draw && !gameOver) {
    statusText.innerHTML = "It's a Draw!";
    statusText.classList.add("draw");
    setInterval(function () {
      statusText.innerHTML += ".";
    }, 1000);
    setTimeout(function () {
      location.reload();
    }, 4000);
  }
}

function game(id) {
  let element = document.getElementById(id);
  if (element.innerHTML === "" && !gameOver) {
    if (turn === "X") {
      element.innerHTML = "X";
      winner();
      if (!gameOver) {
        turn = "O";
        statusText.innerHTML = "Player O's turn";
        setTimeout(makeMoveForO, 500);
      }
    } else {
      console.error("It's not X's turn");
    }
  } else if (element.innerHTML !== "") {
    console.error("Invalid move: Cell is not empty");
  }
}

function makeMoveForO() {
  const board = [];
  for (let i = 1; i <= 9; i++) {
    board.push(
      document.getElementById("item" + i).innerHTML === "X"
        ? 1
        : document.getElementById("item" + i).innerHTML === "O"
        ? -1
        : 0
    );
  }

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ board: board }),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.next_move !== undefined) {
        const nextMove = data.next_move + 1;
        if (document.getElementById("item" + nextMove).innerHTML === "") {
          document.getElementById("item" + nextMove).innerHTML = "O";
          winner();
          turn = "X";
          statusText.innerHTML = "Player X's turn";
        } else {
          console.error("Invalid move by AI: Cell is not empty");
          makeMoveForO();
        }
      } else {
        console.error("Invalid response from server");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
