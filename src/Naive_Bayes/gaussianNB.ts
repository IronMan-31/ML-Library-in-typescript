type Dataset = number[][];
type Labels = number[];

export class GaussianNB {
  private classes: number[] = [];
  private mean: Record<number, number[]> = {};
  private variance: Record<number, number[]> = {};
  private priors: Record<number, number> = {};
  fit(X: Dataset, y: Labels) {
    this.classes = Array.from(new Set(y));

    for (const cls of this.classes) {
      const X_c = X.filter((_, i) => y[i] === cls);
      const n = X_c.length;
      this.priors[cls] = n / X.length;

      const features = X[0].length;
      this.mean[cls] = [];
      this.variance[cls] = [];

      for (let j = 0; j < features; j++) {
        const col = X_c.map((row) => row[j]);
        const mu = col.reduce((a, b) => a + b, 0) / n;
        const varc = col.reduce((a, b) => a + (b - mu) ** 2, 0) / n;

        this.mean[cls].push(mu);
        this.variance[cls].push(varc);
      }
    }
  }
  private gaussianPDF(x: number, mean: number, variance: number): number {
    return (
      Math.exp(-((x - mean) ** 2) / (2 * variance)) /
      Math.sqrt(2 * Math.PI * variance)
    );
  }
  private predictSingle(x: number[]): number {
    let bestClass = this.classes[0];
    let bestProb = -Infinity;

    for (const cls of this.classes) {
      let logProb = Math.log(this.priors[cls]);

      for (let i = 0; i < x.length; i++) {
        logProb += Math.log(
          this.gaussianPDF(x[i], this.mean[cls][i], this.variance[cls][i])
        );
      }

      if (logProb > bestProb) {
        bestProb = logProb;
        bestClass = cls;
      }
    }

    return bestClass;
  }
  predict(X: Dataset): number[] {
    return X.map((x) => this.predictSingle(x));
  }
}
