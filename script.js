window.onresize = changeWindow;
let n = 10;
let grid = [];
for(let i = 0; i < n; i++) {
  let temp = [];
  for(let j = 0; j < n; j++) {
    temp.push(Math.random());
  }
  grid.push(temp);
}
const sqr = Math.pow(n, 2);
let net = new NeuralNetwork(sqr, [sqr, Math.floor(sqr / 2), 2]);
let unit;
let xStart;
let yStart;
let focus = [0, 0];
let variation = 3;
changeWindow();
let curImage = [0, 0];
let middle = Math.floor(n / 2) + .5;

function load() {
  canvas = document.querySelector('.canvas');
  ctx = canvas.getContext('2d');
  canvas.width = width;
  canvas.height = height;
  document.onkeydown = keyPress;
  drawScreen();
  for(let i = 0; i < 1000; i++) {
    netTrain(getData());
  }
}

function runFrame() {
  //DO ALL DRAWING HERE

  requestAnimationFrame(runFrame);
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
      temp.push(getSmallNum());
    }
    newGrid.push(temp);
  }
  for(let i = 1; i < n - 1; i++) {
    newGrid[0][i] = getBigNum();
    newGrid[n - 1][i] = getBigNum();
    newGrid[i][0] = getBigNum();
    newGrid[i][n - 1] = getBigNum();
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
  const guess = net.pass(getInput(grid));
  if(guess[0] > guess[1]) {
    console.log('Net says Cross');
  } else {
    console.log('Net says Circle');
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
