//ACTIVATION FUNCTIONS
const sigmoid = {
  f: z => (1 / (1 + Math.exp(-z))),
  d: z => sigmoid.f(z) * (1 - sigmoid.f(z))
};
const relu = {
  f: z => max(0, z),
  d: z => z < 0? 0:1
};

const colors = ['red', 'green', 'blue'];

class NeuralNetwork {
  //numIn is an int of inputs
  //dim is an array, where the length is the number of layers and
  //the ints inside are the depth of each layer
  //activationFunctions is an array w/ the same length as dim, where
  //a function is an array [func, dx/dy]
  constructor(numIn, dim, params = {}) {
    this.actFunc = params.actFunc || [];
    while(dim.length > this.actFunc.length) {
      this.actFunc.push(sigmoid);
    }
    this.softmax = params.softmax || false;
    this.numIn = numIn;
    this.dim = dim;
    this.weights = [];
    this.biases = [];
    for(let i = 0; i < dim.length; i++) {
      let layer = [];
      let biasLayer = [];
      for(let j = 0; j < dim[i]; j++) {
        biasLayer.push(Math.random() * 2 - 1);
        let node = [];
        const loop = i == 0? numIn : dim[i - 1];
        for(let k = 0; k < loop; k++) {
          node.push(Math.random() * 2 - 1);
        }
        layer.push(node);
      }
      this.weights.push(layer);
      this.biases.push(biasLayer);
    }
    this.error = [[]];
  }

  draw(x, y, w, h) {
    ctx.strokeStyle = 'white';
    let xSpace = w / (this.dim.length + 1);
    let r = h / max(Math.max(...this.dim, this.numIn), this.dim.length + 1) / 4;
    for(let i = 0; i < this.dim.length; i++) {
      let ySpace = h / this.dim[i];
      for(let j = 0; j < this.dim[i]; j++) {
        ctx.beginPath();
        ctx.arc(xSpace * (i + 2) - xSpace / 2, ySpace * (j + 1) - ySpace / 2, r, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
    let ySpace = h / this.numIn;
    for(let j = 0; j < this.numIn; j++) {
      ctx.beginPath();
      ctx.arc(xSpace / 2, ySpace * (j + 1) - ySpace / 2, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  pass(inp) {
    if(inp.length != this.numIn) {
      return false;
    }
    let nextIn = inp;
    for(let l = 0; l < this.weights.length; l++) {
      const layer = this.weights[l];
      const newIn = [];
      for(let j = 0; j < layer.length; j++) {
        const node = layer[j];
        let sum = this.biases[l][j];
        for(let i = 0; i < node.length; i++) {
          sum += node[i] * nextIn[i];
        }
        newIn.push(this.actFunc[l].f(sum));
      }
      nextIn = newIn;
    }
    if(this.softmax) {
      let ret = [];
      let sum = 0;
      for(let n of nextIn) {
        sum += Math.exp(n);
      }
      for(let n of nextIn) {
        ret.push(Math.exp(n) / sum);
      }
      return ret;
    }
    return nextIn;
  }
  drawGraph(x, y, w, h) {
    let n = 0;
    let max = 0; 
    for(let i = 0; i < this.error.length; i++) {
      for(let j = 0; j < this.error[i].length; j++) {
        n++;
        if(this.error[i][j] > max) {
          max = this.error[i][j];
        }
      }
    }
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x + w, y + h);
    ctx.lineTo(x + w, y);
    ctx.lineTo(x, y);
    ctx.stroke();
    let clr = 0;
    let cur = 0;
    for(let i = 0; i < this.error.length; i++) {
      ctx.strokeStyle = colors[clr];
      for(let j = 0; j < this.error[i].length; j++) {
        let pointX = (cur / n * w) + x;
        let pointY = ((1 - (this.error[i][j] / max)) * h) + y;
        ctx.strokeRect(pointX, pointY, 1, 1);
        cur++;
      }
      clr++;
      clr %= colors.length;
    }
  }

  getGradient(inp, ans) {
    const weightGradient = JSON.parse(JSON.stringify(this.weights));
    const biasGradient = JSON.parse(JSON.stringify(this.biases));
    let inputs = [];
    let outputs = [];
    let nextIn = inp;
    //Pass the input
    for(let i = 0; i < this.weights.length; i++) {
      const temp = [];
      let inpTemp = [];
      let outTemp = [];
      for(let j = 0; j < this.weights[i].length; j++) {
        let sum = this.biases[i][j];
        for(let k = 0; k < this.weights[i][j].length; k++) {
          sum += this.weights[i][j][k] * nextIn[k];
        }
        let out = this.actFunc[i].f(sum);
        inpTemp.push(sum);
        temp.push(out);
        outTemp.push(out);
      }
      nextIn = temp;
      inputs.push(inpTemp);
      outputs.push(outTemp);
    }
    
    let predicted = nextIn;
    const finalInd = outputs.length - 1;
    
    if(this.softmax) {
      predicted = [];
      let sum = 0;
      for(let n of nextIn) {
        sum += Math.exp(n);
      }
      for(let n of nextIn) {
        predicted.push(Math.exp(n) / sum);
      }
    }

    let error = 0;
    for(let i = 0; i < predicted.length; i++) {
      error += Math.pow(ans[i] - predicted[i], 2);
    }
    this.error[this.error.length - 1].push(error);

    //Get derivative of last layer
    for(let i = 0; i < outputs[finalInd].length; i++) {
      if(this.softmax) {
        let sum = 0;
        for(let j = 0; j < predicted.length; j++) {
          if(i == j) {
            sum += predicted[j] * (1 - outputs[finalInd][i]) * -2 * (ans[j] - predicted[j]);
          } else {
            sum += predicted[j] * (0 - outputs[finalInd][i]) * -2 * (ans[j] - predicted[j]);
          }
        }
        outputs[finalInd][i] = sum;
      } else {
        outputs[finalInd][i] = -2 * (ans[i] - predicted[i]);
      }
    }

    for(let i = finalInd; i >= 0; i--) {
      //Set output derivative if it's not the first one
      if(i != finalInd) {
        //Go through the outputs of the layer
        for(let j = 0; j < outputs[i].length; j++) {
          //Add up the inputs of the layer in front
          let sum = 0;
          for(let k = 0; k < inputs[i + 1].length; k++) {
            //input in front * Weight that connects inp (k) to current node (j)
            sum += inputs[i + 1][k] * this.weights[i + 1][k][j];
          }
          outputs[i][j] = sum;
        }
      }
      
      for(let j = 0; j < outputs[i].length; j++) {
        inputs[i][j] = outputs[i][j] * this.actFunc[i].d(inputs[i][j]);
      }

      let previousOutputs;
      if(i == 0) {
        previousOutputs = inp;
      } else {
        previousOutputs = outputs[i - 1];
      }
      //For each node in layer
      for(let j = 0; j < this.weights[i].length; j++) {
        //Go through each weight and find the derivative
        for(let k = 0; k < this.weights[i][j].length; k++) {
          //i = Layer   j = Node in layer   k = Node in previous Layer
          weightGradient[i][j][k] = inputs[i][j] * previousOutputs[k];
        }
      }

    }
    return [weightGradient, inputs];
  }

  //Inp is a 2d Array of all data
  //Ans is also a 2d Array
  trainBatch(inp, ans, lr) {
    let first = this.getGradient(inp[0], ans[0]);
    let fullWeightGradient = first[0];
    let fullBiasGradient = first[1];
    for(let i = 1; i < inp.length; i++) {
      const gradient = this.getGradient(inp[i], ans[i]);
      let weightGradient = gradient[0];
      const biasGradient = gradient[1];
      for(let x = 0; x < fullWeightGradient.length; x++) {
        for(let y = 0; y < fullWeightGradient[x].length; y++) {
          fullBiasGradient[x][y] += biasGradient[x][y];
          for(let z = 0; z < fullWeightGradient[x][y].length; z++) {
            fullWeightGradient[x][y][z] += weightGradient[x][y][z];
          }
        }
      }
    }
    for(let i = 0; i < fullWeightGradient.length; i++) {
      for(let j = 0; j < fullWeightGradient[i].length; j++) {
        this.biases[i][j] -= fullBiasGradient[i][j] / inp.length * lr;
        for(let k = 0; k < fullWeightGradient[i][j].length; k++) {
          let oldWeight = this.weights[i][j][k];
          this.weights[i][j][k] -= fullWeightGradient[i][j][k] / inp.length * lr;
        }
      }
    }
    this.error.push([]);
  }
}
