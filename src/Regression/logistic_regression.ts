import { Accuracy, BinaryCrossEntropy } from "../../metrics";

export class LogisticRegression {
  learning_rate: number;
  weight: number[];
  bias: number;
  iterations: number;
  constructor(learning_rate: number, iterations: number = 100) {
    this.learning_rate = learning_rate;
    this.iterations = iterations;
    this.bias = Math.random();
    this.weight = [];
  }
  get_output(
    train_data: number[][],
    weights: number[],
    bias: number
  ): number[] {
    let siz = train_data.length;
    let ans: number[] = [];
    let temp = 0;
    for (let i = 0; i < siz; i += 1) {
      temp = 0;
      for (let j = 0; j < weights.length; j += 1) {
        temp += weights[j] * train_data[i][j];
      }
      temp += bias;
      const eps = 1e-15;
      temp = 1 / (1 + Math.exp(-temp));
      temp = Math.min(Math.max(temp, eps), 1 - eps);
      ans.push(temp);
      temp = 0;
    }
    return ans;
  }

  private initialise_weights(train_data: number[][]): void {
    let siz: number = train_data[0].length;
    for (let i = 0; i < siz; i += 1) {
      this.weight.push(Math.random() * 0.1);
    }
  }

  private get_gradients_weights(
    train_data: number[][],
    y_pred: number[],
    y_true: number[],
    id: number
  ): number {
    let ans = 0;
    for (let i = 0; i < y_true.length; i += 1) {
      ans += train_data[i][id] * (y_pred[i] - y_true[i]);
    }
    return (1 / y_true.length) * ans;
  }

  private get_gradients_bias(y_pred: number[], y_true: number[]): number {
    let ans = 0;
    for (let i = 0; i < y_true.length; i += 1) {
      ans += y_pred[i] - y_true[i];
    }
    return (1 / y_true.length) * ans;
  }

  private gradient_descent(
    train_data: number[][],
    y_true: number[],
    y_pred: number[]
  ): void {
    for (let i = 0; i < this.weight.length; i += 1) {
      this.weight[i] =
        this.weight[i] -
        this.learning_rate *
          this.get_gradients_weights(train_data, y_pred, y_true, i);
    }
    this.bias =
      this.bias - this.learning_rate * this.get_gradients_bias(y_pred, y_true);
  }

  fit(train_data: number[][], train_output: number[]): void {
    if (train_data.length == 0) {
      throw new Error("Empty data was passed");
    }
    if (train_data.length != train_output.length) {
      throw new Error("Train Input and Output don't have equal dimensions");
    }
    this.initialise_weights(train_data);
    for (let i = 0; i < this.iterations; i += 1) {
      let y_pred: number[] = this.get_output(
        train_data,
        this.weight,
        this.bias
      );
      let error = BinaryCrossEntropy(train_output, y_pred);
      let acc = Accuracy(train_output, y_pred);
      console.log(`Error : ${error} -- Accuracy : ${acc}`);
      this.gradient_descent(train_data, train_output, y_pred);
    }
  }
  predict(train_data: number[][]): number[] {
    const probs = this.get_output(train_data, this.weight, this.bias);
    return probs.map((p) => (p >= 0.5 ? 1 : 0));
  }
}
