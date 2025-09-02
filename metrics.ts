export function MeanSquaredError(
  y_true: number[],
  y_pred: number[],
  squared: boolean = true
): number {
  if (y_true.length != y_pred.length) {
    throw new Error("y_true and y_pred have different dimensions");
  }
  if (y_true.length == 0) {
    throw new Error("y_true and y_pred are empty");
  }
  let ans = 0;
  for (let i = 0; i < y_true.length; i += 1) {
    ans += (y_true[i] - y_pred[i]) ** 2;
  }

  return squared ? ans / y_true.length : (ans / y_true.length) ** 0.5;
}

export function BinaryCrossEntropy(y_true: number[], y_pred: number[]): number {
  if (y_true.length != y_pred.length) {
    throw new Error("y_true and y_pred have different dimensions");
  }
  if (y_true.length == 0) {
    throw new Error("y_true and y_pred are empty");
  }
  let ans = 0;
  const eps = 1e-15;
  for (let i = 0; i < y_true.length; i += 1) {
    const y_hat = Math.min(Math.max(y_pred[i], eps), 1 - eps);
    ans += y_true[i] * Math.log(y_hat) + (1 - y_true[i]) * Math.log(1 - y_hat);
  }
  return -ans / y_true.length;
}

export function Accuracy(y_true: number[], y_pred: number[]): number {
  let ans = 0;
  let y_hat: number[] = y_pred.map((val) => {
    return val >= 0.5 ? 1 : 0;
  });
  for (let i = 0; i < y_true.length; i += 1) {
    if (y_true[i] == y_hat[i]) {
      ans += 1;
    }
  }
  return ans / y_true.length;
}
function get_random_slices(length: number, count: number): number[] {
  const indices = new Set<number>();
  while (indices.size < count) {
    indices.add(Math.floor(Math.random() * length));
  }
  return Array.from(indices);
}

export function train_test_split(
  train_data: number[][],
  split_ratio: number = 0.2
): number[][][] {
  let indices = get_random_slices(
    train_data.length,
    Math.floor(train_data.length * (1 - split_ratio))
  );
  let train_dt = train_data.filter((val, id) => {
    return indices.includes(id);
  });
  let val_dt = train_data.filter((val, id) => {
    return !indices.includes(id);
  });
  return [train_dt, val_dt];
}

export function KFold(data: number[][], folds: number = 5) {
  const shuffled = [...data].sort(() => Math.random() - 0.5);
  const foldSize = Math.floor(data.length / folds);
  const output: number[][][][] = [];

  for (let i = 0; i < folds; i++) {
    const val_dt = shuffled.slice(i * foldSize, (i + 1) * foldSize);
    const train_dt = [
      ...shuffled.slice(0, i * foldSize),
      ...shuffled.slice((i + 1) * foldSize),
    ];
    output.push([train_dt, val_dt]);
  }
  return output;
}
