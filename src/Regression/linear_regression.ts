import { MeanSquaredError } from "../../metrics";

export class LinearRegression {
  learning_rate: number;
  weight: number[];
  bias: number;
  iterations: number;

  constructor(learning_rate: number, iterations: number = 100) {
    this.learning_rate = learning_rate;
    this.iterations = iterations;
    this.bias = 0;
    this.weight = [];
  }

  get_output(
    train_data: number[][],
    weights: number[] = this.weight,
    bias: number = this.bias
  ): number[] {
    return train_data.map((row) => {
      return row.reduce((sum, val, j) => sum + weights[j] * val, bias);
    });
  }

  private initialise_weights(train_data: number[][]): void {
    let siz: number = train_data[0].length;
    this.weight = [];
    for (let i = 0; i < siz; i += 1) {
      this.weight.push(Math.random() * 0.01);
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
    return (2 / y_true.length) * ans;
  }

  private get_gradients_bias(y_pred: number[], y_true: number[]): number {
    let ans = 0;
    for (let i = 0; i < y_true.length; i += 1) {
      ans += y_pred[i] - y_true[i];
    }
    return (2 / y_true.length) * ans;
  }

  private gradient_descent(
    train_data: number[][],
    y_true: number[],
    y_pred: number[]
  ): void {
    for (let i = 0; i < this.weight.length; i += 1) {
      this.weight[i] -=
        this.learning_rate *
        this.get_gradients_weights(train_data, y_pred, y_true, i);
    }
    this.bias -= this.learning_rate * this.get_gradients_bias(y_pred, y_true);
  }

  fit(train_data: number[][], train_output: number[]): void {
    if (train_data.length == 0) throw new Error("Empty data was passed");
    if (train_data.length != train_output.length)
      throw new Error("Train Input and Output don't match");

    this.initialise_weights(train_data);

    for (let i = 0; i < this.iterations; i += 1) {
      let y_pred = this.get_output(train_data, this.weight, this.bias);
      let error = MeanSquaredError(train_output, y_pred);

      if (i % 10 === 0) console.log(`Iteration ${i}: MSE = ${error}`);

      this.gradient_descent(train_data, train_output, y_pred);
    }
  }
}
