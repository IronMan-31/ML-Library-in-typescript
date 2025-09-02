import {
  DecisionTreeRegressor,
  Node,
} from "../Decision_Tree/decision_tree_regressor";

export class RandomForestRegressor {
  max_depth: number;
  n_estimators: number;
  max_features: number;
  min_samples_split: number;
  max_samples: number;

  constructor(
    labels: number,
    n_estimators: number = 100,
    max_depth: number = Infinity,
    max_features: number = Infinity,
    max_samples: number = Infinity,
    min_samples_split: number = 2
  ) {
    this.max_depth = max_depth;
    this.n_estimators = n_estimators;
    this.max_features = max_features;
    this.min_samples_split = min_samples_split;
    this.max_samples = max_samples;
  }

  private get_random_indices(length: number, count: number): number[] {
    const indices: number[] = [];
    for (let i = 0; i < count; i++) {
      indices.push(Math.floor(Math.random() * length));
    }
    return indices;
  }

  private train_single_tree(
    train_data: number[][],
    train_output: number[]
  ): [Node, DecisionTreeRegressor] {
    const sample_count =
      this.max_samples === Infinity ? train_data.length : this.max_samples;
    const indices = this.get_random_indices(train_data.length, sample_count);

    let temporary_data = train_data.filter((_, id) => indices.includes(id));
    let temporary_output = train_output.filter((_, id) => indices.includes(id));

    if (this.max_features !== Infinity) {
      if (this.max_features > train_data[0].length)
        throw new Error(
          "Max features should be less than given number of columns"
        );
      const features = this.get_random_indices(
        train_data[0].length,
        this.max_features
      );
      temporary_data = temporary_data.map((row) =>
        row.filter((_, id) => features.includes(id))
      );
    }

    const tree = new DecisionTreeRegressor(
      this.max_depth,
      this.min_samples_split
    );
    const trained_tree = tree.fit(temporary_data, temporary_output);

    return [trained_tree, tree];
  }

  fit(
    train_data: number[][],
    train_output: number[]
  ): [Node, DecisionTreeRegressor][] {
    if (train_data.length == 0) {
      throw new Error("Empty Dataset was passed");
    }
    if (train_data[0].length == train_output.length) {
      throw new Error("Dimension mismatch in input and output");
    }
    const forest: [Node, DecisionTreeRegressor][] = [];
    for (let i = 0; i < this.n_estimators; i++) {
      forest.push(this.train_single_tree(train_data, train_output));
    }
    return forest;
  }

  get_single_output(
    forest: [Node, DecisionTreeRegressor][],
    sample: number[]
  ): number {
    let outputs: number[] = [];
    for (let i = 0; i < forest.length; i += 1) {
      outputs.push(forest[i][1].get_single_output(forest[i][0], sample));
    }
    let ans = 0;
    for (let i = 0; i < outputs.length; i += 1) {
      ans += outputs[i];
    }
    return ans / outputs.length;
  }

  predict(
    forest: [Node, DecisionTreeRegressor][],
    val_data: number[][]
  ): number[] {
    return val_data.map((val) => {
      return this.get_single_output(forest, val);
    });
  }
}
