export interface Node {
  left?: Node;
  feature?: number;
  threshold?: number | string | null;
  right?: Node;
  value?: number;
}

export class DecisionTreeClassifier {
  different_labels: number;
  max_depth: number;
  min_samples_split: number;
  constructor(
    labels: number,
    max_depth: number = Infinity,
    min_samples_split: number = 2
  ) {
    this.different_labels = labels;
    this.max_depth = max_depth;
    this.min_samples_split = min_samples_split;
  }

  private calc_gini(labels: number[]): number {
    if (labels.length === 0) return 0;
    let obj: Record<number, number> = {};
    for (let i = 0; i < labels.length; i += 1) {
      obj[labels[i]] = (obj[labels[i]] || 0) + 1;
    }
    let gini = 1;
    for (const key in obj) {
      const prob = obj[key] / labels.length;
      gini -= prob * prob;
    }
    return gini;
  }

  private ginidecrease(
    parent_labels: number[],
    left_labels: number[],
    right_labels: number[]
  ): number {
    const parent_gini = this.calc_gini(parent_labels);
    const left_gini = this.calc_gini(left_labels);
    const right_gini = this.calc_gini(right_labels);
    const weightedGini =
      (left_labels.length / parent_labels.length) * left_gini +
      (right_labels.length / parent_labels.length) * right_gini;
    return parent_gini - weightedGini;
  }

  private treat_num_cols(
    train_data: (number | string)[][],
    train_output: number[],
    id: number
  ): [number, number] {
    const colWithLabels = train_data.map((row, i) => ({
      value: row[id] as number,
      label: train_output[i],
    }));
    colWithLabels.sort((a, b) => (a.value as number) - (b.value as number));

    let maxGiniDec = 0;
    let bestThreshold = colWithLabels[0].value as number;

    for (let i = 1; i < colWithLabels.length; i++) {
      if (colWithLabels[i].label !== colWithLabels[i - 1].label) {
        const threshold =
          ((colWithLabels[i].value as number) +
            (colWithLabels[i - 1].value as number)) /
          2;

        const leftLabels: number[] = [];
        const rightLabels: number[] = [];
        for (const item of colWithLabels) {
          if ((item.value as number) <= threshold) leftLabels.push(item.label);
          else rightLabels.push(item.label);
        }

        const ginDc = this.ginidecrease(train_output, leftLabels, rightLabels);

        if (ginDc > maxGiniDec) {
          maxGiniDec = ginDc;
          bestThreshold = threshold;
        }
      }
    }

    return [bestThreshold, maxGiniDec];
  }

  private treat_cat_cols(
    train_data: (number | string)[][],
    train_output: number[],
    id: number
  ): [string, number] {
    const diff_labels: Record<string, number> = {};
    for (let i = 0; i < train_data.length; i += 1) {
      const key = String(train_data[i][id]);
      diff_labels[key] = (diff_labels[key] || 0) + 1;
    }
    let maxGiniDec = 0;
    let ans = "";
    for (const key in diff_labels) {
      let left_labels: number[] = [];
      let right_labels: number[] = [];
      for (let i = 0; i < train_output.length; i += 1) {
        if (String(train_data[i][id]) === key) {
          left_labels.push(train_output[i]);
        } else {
          right_labels.push(train_output[i]);
        }
      }
      let ginDc = this.ginidecrease(train_output, left_labels, right_labels);
      if (ginDc > maxGiniDec) {
        maxGiniDec = ginDc;
        ans = key;
      }
    }
    return [ans, maxGiniDec];
  }

  private treat_all_cols(
    train_data: (number | string)[][],
    train_output: number[]
  ) {
    let maxGinDc = 0;
    let ans: number | string = 0;
    let id = -1;
    for (let i = 0; i < train_data[0].length; i += 1) {
      let temp: [string | number, number];
      if (typeof train_data[0][i] === "string") {
        temp = this.treat_cat_cols(train_data, train_output, i);
      } else {
        temp = this.treat_num_cols(train_data, train_output, i);
      }
      if (temp[1] > maxGinDc) {
        ans = temp[0];
        id = i;
        maxGinDc = temp[1];
      }
    }
    let left_side_data: (number | string)[][] = [];
    let left_side_output: number[] = [];
    let right_side_data: (number | string)[][] = [];
    let right_side_output: number[] = [];

    for (let i = 0; i < train_data.length; i += 1) {
      if (train_data[i][id] === ans) {
        left_side_data.push(train_data[i]);
        left_side_output.push(train_output[i]);
      } else {
        right_side_data.push(train_data[i]);
        right_side_output.push(train_output[i]);
      }
    }

    return [
      left_side_data,
      left_side_output,
      right_side_data,
      right_side_output,
      id,
      ans,
    ] as [
      (number | string)[][],
      number[],
      (number | string)[][],
      number[],
      number,
      number | string
    ];
  }

  fit(
    train_data: (number | string)[][],
    train_output: number[],
    depth: number = 0
  ): Node {
    const uniqueLabels = Array.from(new Set(train_output));

    if (uniqueLabels.length === 1) {
      return { value: uniqueLabels[0] };
    }

    if (
      train_data[0].length === 0 ||
      train_output.length < this.min_samples_split ||
      depth >= this.max_depth
    ) {
      const counts: Record<number, number> = {};
      train_output.forEach((val) => (counts[val] = (counts[val] || 0) + 1));
      const majority = Number(
        Object.keys(counts).reduce((a, b) =>
          counts[Number(a)] > counts[Number(b)] ? a : b
        )
      );
      return { value: majority };
    }

    const [
      leftData,
      leftLabels,
      rightData,
      rightLabels,
      bestFeatureIndex,
      bestThreshold,
    ] = this.treat_all_cols(train_data, train_output);

    if (leftData.length === 0 || rightData.length === 0) {
      const counts: Record<number, number> = {};
      train_output.forEach((val) => (counts[val] = (counts[val] || 0) + 1));
      const majority = Number(
        Object.keys(counts).reduce((a, b) =>
          counts[Number(a)] > counts[Number(b)] ? a : b
        )
      );
      return { value: majority };
    }
    const node: Node = {
      feature: bestFeatureIndex,
      threshold: bestThreshold,
      left: this.fit(leftData, leftLabels, depth + 1),
      right: this.fit(rightData, rightLabels, depth + 1),
    };
    return node;
  }

  get_single_output(
    node: Node,
    sample: (string | number)[]
  ): number | undefined {
    if (node.value !== undefined) {
      return node.value;
    }

    if (typeof node.threshold === "number") {
      if ((sample[node.feature!] as number) < node.threshold) {
        return this.get_single_output(node.left!, sample);
      } else {
        return this.get_single_output(node.right!, sample);
      }
    } else {
      if (sample[node.feature!] === node.threshold) {
        return this.get_single_output(node.left!, sample);
      } else {
        return this.get_single_output(node.right!, sample);
      }
    }
  }

  predict(tree: Node, val_data: (number | string)[][]): (number | undefined)[] {
    return val_data.map((val) => {
      return this.get_single_output(tree, val);
    });
  }
}
