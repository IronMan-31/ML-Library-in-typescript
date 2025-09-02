export function dotProduct(vec1: number[], vec2: number[]): number {
  let ans = 0;
  if (vec1.length != vec2.length) {
    throw new Error("Size mismatch between Vectors");
  }
  let vector1Pointer = 0;
  let vector2Pointer = 0;
  while (vector1Pointer < vec1.length) {
    ans += vec1[vector1Pointer] * vec2[vector2Pointer];
    vector1Pointer += 1;
    vector2Pointer += 1;
  }
  return ans;
}

export function matMul(A: number[][], B: number[][]): number[][] {
  if (A.length === 0 || B.length === 0) {
    throw new Error("Empty matrix passed");
  }

  const rowsA = A.length;
  const colsA = A[0].length;
  const rowsB = B.length;
  const colsB = B[0].length;

  if (colsA !== rowsB) {
    throw new Error("Dimension mismatch: cannot multiply");
  }

  const result: number[][] = Array.from({ length: rowsA }, () =>
    Array(colsB).fill(0)
  );

  for (let i = 0; i < rowsA; i++) {
    for (let j = 0; j < colsB; j++) {
      for (let k = 0; k < colsA; k++) {
        result[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  return result;
}

export function Transpose(A: number[][]): number[][] {
  if (A.length === 0) return [];
  let ans: number[][] = [];
  for (let i = 0; i < A[0].length; i += 1) {
    let temp: number[] = [];
    for (let j = 0; j < A.length; j += 1) {
      temp.push(A[j][i]);
    }
    ans.push(temp);
  }
  return ans;
}

export function mean(A: number[][]): number {
  let ans: number = 0;
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < A[0].length; j += 1) {
      ans += A[i][j];
    }
  }
  return ans / (A.length * A[0].length);
}

export function std(A: number[][]): number {
  let mn = mean(A);
  let ans = 0;
  for (let i = 0; i < A.length; i += 1) {
    for (let j = 0; j < A[0].length; j += 1) {
      ans += (A[i][j] - mn) ** 2;
    }
  }
  return (ans / (A.length * A[0].length)) ** 0.5;
}

export function determinant(A: number[][]): number {
  if (A.length === 0) {
    throw new Error("Empty matrix passed");
  }
  if (A.length !== A[0].length) {
    throw new Error("Dimension mismatch: must be a square matrix");
  }
  const n = A.length;
  if (n === 1) return A[0][0];
  if (n === 2) return A[0][0] * A[1][1] - A[0][1] * A[1][0];
  let ans = 0;
  for (let j = 0; j < n; j++) {
    const minor = A.slice(1).map((row) => row.filter((_, col) => col !== j));
    ans += (-1) ** j * A[0][j] * determinant(minor);
  }
  return ans;
}

export function Inverse(A: number[][]): number[][] {
  if (A.length == 0) {
    throw new Error("Empty Matrix was passed");
  }
  if (A.length != A[0].length) {
    throw new Error("Dimension Mismatch: Only square matrix can have Inverse");
  }
  let det = determinant(A);
  if (det == 0) {
    throw new Error("Determinant is 0");
  }
  if (A.length === 2) {
    return [
      [A[1][1] / det, -A[0][1] / det],
      [-A[1][0] / det, A[0][0] / det],
    ];
  }
  const cofactors: number[][] = [];
  for (let i = 0; i < A.length; i++) {
    cofactors[i] = [];
    for (let j = 0; j < A.length; j++) {
      const minor = A.filter((_, row) => row !== i).map((row) =>
        row.filter((_, col) => col !== j)
      );

      cofactors[i][j] = (-1) ** (i + j) * determinant(minor);
    }
  }
  const adjugate = cofactors[0].map((_, j) => cofactors.map((row) => row[j]));
  return adjugate.map((row) => row.map((val) => val / det));
}
