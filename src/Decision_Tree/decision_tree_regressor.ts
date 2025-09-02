export interface Node {
  left?: Node;
  feature?: number;
  threshold?: number | string | null;
  right?: Node;
  value?: number;
}

export class DecisionTreeRegressor {
  max_depth: number;
  min_samples_split: number;

  constructor(max_depth: number = Infinity, min_samples_split: number = 2) {
    this.max_depth = max_depth;
    this.min_samples_split = min_samples_split;
  }

  private calc_variance(y: number[]): number {
    if (y.length === 0) return 0;
    const mean = y.reduce((a, b) => a + b, 0) / y.length;
    return y.reduce((sum, val) => sum + (val - mean) ** 2, 0) / y.length;
  }

  private variance_decrease(
    parent: number[],
    left: number[],
    right: number[]
  ): number {
    const parent_var = this.calc_variance(parent);
    const weighted_var =
      (left.length / parent.length) * this.calc_variance(left) +
      (right.length / parent.length) * this.calc_variance(right);
    return parent_var - weighted_var;
  }

  private treat_num_cols(
    data: number[][],
    target: number[],
    id: number
  ): [number, number] {
    const colWithTarget = data.map((row, i) => ({
      value: row[id],
      target: target[i],
    }));
    colWithTarget.sort((a, b) => a.value - b.value);

    let bestThreshold = colWithTarget[0].value;
    let maxVarDec = 0;

    for (let i = 1; i < colWithTarget.length; i++) {
      if (colWithTarget[i].target !== colWithTarget[i - 1].target) {
        const threshold =
          (colWithTarget[i].value + colWithTarget[i - 1].value) / 2;

        const left: number[] = [];
        const right: number[] = [];
        colWithTarget.forEach((item) => {
          if (item.value <= threshold) {
            left.push(item.target);
          } else {
            right.push(item.target);
          }
        });

        const varDec = this.variance_decrease(target, left, right);
        if (varDec > maxVarDec) {
          maxVarDec = varDec;
          bestThreshold = threshold;
        }
      }
    }

    return [bestThreshold, maxVarDec];
  }

  private treat_all_cols(
    data: number[][],
    target: number[]
  ): [number[][], number[], number[][], number[], number, number] {
    let maxVarDec = 0;
    let bestFeature = -1;
    let bestThreshold = 0;

    for (let i = 0; i < data[0].length; i++) {
      const [threshold, varDec] = this.treat_num_cols(data, target, i);
      if (varDec > maxVarDec) {
        maxVarDec = varDec;
        bestFeature = i;
        bestThreshold = threshold;
      }
    }

    const leftData: number[][] = [];
    const leftTarget: number[] = [];
    const rightData: number[][] = [];
    const rightTarget: number[] = [];

    for (let i = 0; i < data.length; i++) {
      if (data[i][bestFeature] <= bestThreshold) {
        leftData.push(data[i]);
        leftTarget.push(target[i]);
      } else {
        rightData.push(data[i]);
        rightTarget.push(target[i]);
      }
    }

    return [
      leftData,
      leftTarget,
      rightData,
      rightTarget,
      bestFeature,
      bestThreshold,
    ];
  }

  fit(data: number[][], target: number[], depth: number = 0): Node {
    if (target.length < this.min_samples_split || depth >= this.max_depth) {
      const meanVal = target.reduce((a, b) => a + b, 0) / target.length;
      return { value: meanVal };
    }

    if (data[0].length === 0) {
      const meanVal = target.reduce((a, b) => a + b, 0) / target.length;
      return { value: meanVal };
    }

    const [
      leftData,
      leftTarget,
      rightData,
      rightTarget,
      bestFeature,
      bestThreshold,
    ] = this.treat_all_cols(data, target);

    if (leftData.length === 0 || rightData.length === 0) {
      const meanVal = target.reduce((a, b) => a + b, 0) / target.length;
      return { value: meanVal };
    }

    return {
      feature: bestFeature,
      threshold: bestThreshold,
      left: this.fit(leftData, leftTarget, depth + 1),
      right: this.fit(rightData, rightTarget, depth + 1),
    };
  }

  get_single_output(node: Node, sample: number[]): number {
    if (node.value !== undefined) return node.value;
    if (sample[node.feature]! <= node.threshold!) {
      return this.get_single_output(node.left!, sample);
    } else {
      return this.get_single_output(node.right!, sample);
    }
  }

  predict(tree: Node, val_data: number[][]): number[] {
    return val_data.map((row) => this.get_single_output(tree, row));
  }
}
