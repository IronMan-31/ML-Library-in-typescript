export class KNearestNeighours {
  k: number;
  train_data: number[][];
  train_output: number[];
  constructor(k: number = 5) {
    this.k = k;
  }
  private euclidean_distance(vec1: number[], vec2: number[]): number {
    let ans = 0;
    if (vec1.length != vec2.length) {
      throw new Error("Train_data and val_data should have same columns");
    }
    for (let i = 0; i < vec1.length; i += 1) {
      ans += (vec1[i] - vec2[i]) ** 2;
    }
    return Math.sqrt(ans);
  }
  private get_Nearest_neighours(
    train_data: number[][],
    train_output: number[],
    vec1: number[]
  ): number[] {
    let ans: number[][] = [];
    for (let i = 0; i < train_data.length; i += 1) {
      ans.push([this.euclidean_distance(train_data[i], vec1), train_output[i]]);
    }
    ans.sort((a, b) => a[0] - b[0]);
    let ans1 = ans.slice(0, this.k).map((val) => val[1]);
    return ans1;
  }

  private get_output(
    train_data: number[][],
    train_output: number[],
    vec1: number[]
  ): number {
    const neighbors = this.get_Nearest_neighours(
      train_data,
      train_output,
      vec1
    );

    const counts: Record<number, number> = {};
    let maxCount = 0;
    let predicted = -1;

    for (const label of neighbors) {
      counts[label] = (counts[label] || 0) + 1;
      if (counts[label] > maxCount) {
        maxCount = counts[label];
        predicted = label;
      }
    }

    return predicted;
  }

  fit(train_data: number[][], train_output: number[]): void {
    if (train_data.length == 0) {
      throw new Error("Empty Data was passed");
    }
    if (train_data.length != train_output.length) {
      throw new Error("Dimension Mismatch");
    }
    if (train_data.length < this.k) {
      throw new Error("Training example are less than given k");
    }
    this.train_data = train_data;
    this.train_output = train_output;
  }
  predict(val_data: number[][]): number[] {
    let ans: number[] = [];
    for (let i = 0; i < val_data.length; i += 1) {
      ans.push(
        this.get_output(this.train_data, this.train_output, val_data[i])
      );
    }
    return ans;
  }
}
