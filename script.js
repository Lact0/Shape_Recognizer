window.onresize = changeWindow;
let n = 10;
let grid;
let curImage = [0, 0];
const sqr = Math.pow(n, 2);
let net = new NeuralNetwork(sqr, [sqr, Math.floor(sqr / 2), 2]);
let unit;
let xStart;
let yStart;
let focus = [0, 0];
let variation = 1.3;
changeWindow();
let mid = Math.floor(n / 2) + (.5 * (n % 2 - 1));
grid = makeCircle();
let text;

function load() {
  canvas = document.querySelector('.canvas');
  ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  document.onkeydown = keyPress;
  text = document.getElementById("topText");
  drawScreen();
  for(let i = 0; i < 1000; i++) {
    netTrain(getData());
  }
  netGuess();
}

function getData() {
  if(Math.random() > .5) {
    return [getInput(makeCross()), [1, 0]];
  } else {
    return [getInput(makeCircle()), [0, 1]];
  }
}

function getSmallNum() {
  return Math.random() / variation;
}
function getBigNum() {
  return getSmallNum() + (1 / variation) * (variation - 1);
}

function makeCross() {
  let newGrid = [];
  for(let i = 0; i < n; i++) {
    let temp = [];
    for(let j = 0; j < n; j++) {
      if(i == j || j == (n - i - 1)) {
        temp.push(getBigNum());
      } else {
        temp.push(getSmallNum());
      }
    }
    newGrid.push(temp);
  }
  return newGrid;
}

function makeCircle() {
  let newGrid = [];
  for(let i = 0; i < n; i++) {
    let temp = [];
    for(let j = 0; j < n; j++) {
      let dist = Math.sqrt(Math.pow(i - mid, 2) + Math.pow(j - mid, 2));
      dist -= Math.floor(n / 2) - .5;
      dist = Math.abs(dist);
      if(dist <= .5) {
        temp.push(getBigNum());
      } else {
        temp.push(getSmallNum());
      }
      
    }
    newGrid.push(temp);
  }
  return newGrid;
}

function getInput(gr) {
  let inputs = [];
  for(let i = 0; i < gr.length; i++) {
    for(let j = 0; j < gr[i].length; j++) {
      inputs.push(gr[i][j]);
    }
  }
  return inputs;
}

function netGuess() {
  let guess = net.pass(getInput(grid));
  guess = guess[0] > guess[1]? 1 : 0;
  let ans;
  if(guess == curImage[0]) {
    ans = 'Correct!'
  } else {
    ans = 'Wrong!';
  }
  if(guess == 1) {
    text.innerHTML = 'Net Guess: Cross - ' + ans;
  } else {
    text.innerHTML = 'Net Guess: Circle - ' + ans;
  }
}

function netTrain(data) {
  net.trainBatch([data[0]], [data[1]], .1);
}

function drawScreen() {
  ctx.clearRect(0, 0, width, height);
  for(let i = 0; i < n; i++) {
    for(let j = 0; j < n; j++) {
      const color = ((1 - grid[i][j]) * 255) - 20;
      ctx.fillStyle = 'rgba(' + color + ', ' + color + ', ' + color + ', 1)';
      ctx.fillRect(xStart + unit * i, yStart + unit * j, unit, unit);
    }
  }
  ctx.strokeStyle = 'white';
  for(let i = 0; i < n; i++) {
    for(let j = 0; j < n; j++) {
      ctx.lineWidth = 2;
      if(focus[0] == i && focus[1] == j) {
        ctx.lineWidth = 6;
      }
      ctx.strokeRect(xStart + unit * i, yStart + unit * j, unit, unit);
    }
  }
}

function changeWindow() {
  width = window.innerWidth;
  height = window.innerHeight;
  unit = min(width, height) / (n + 1);
  xStart = (width - (unit * n)) / 2;
  yStart = (height - (unit * n)) / 2;
  //REDRAW SCREEN
}

function keyPress(key) {
  if(key.keyCode == 32) {
    let rand = Math.random();
    if(rand > .5) {
      grid = makeCross();
      curImage = [1, 0];
    } else {
      grid = makeCircle();
      curImage = [0, 1];
    }
    netGuess();
  }
  if(key.keyCode == 38) {
    grid[focus[0]][focus[1]] += .05;
  }
  if(key.keyCode == 40) {
    grid[focus[0]][focus[1]] -= .05;
  }
  grid[focus[0]][focus[1]] = min(max(grid[focus[0]][focus[1]], 0), 1);
  drawScreen();
}

function leftClick() {
  const x = event.clientX;
  const y = event.clientY;
  let xGridPos = parseInt((x - xStart) / unit);
  let yGridPos = parseInt((y - yStart) / unit);
  xGridPos = min(max(xGridPos, 0), n - 1);
  yGridPos = min(max(yGridPos, 0), n - 1);
  focus = [xGridPos, yGridPos];
  drawScreen();
}
