import { DecisionTreeClassifier } from "./Decision_Tree/decision_tree_classifier";
import { DecisionTreeRegressor } from "./Decision_Tree/decision_tree_regressor";
import { KNearestNeighours } from "./KNN/knn";
import { RandomForestClassifier } from "./Random_Forest/random_forest_classifier";
import { RandomForestRegressor } from "./Random_Forest/random_forest_regressor";
import { LinearRegression } from "./Regression/linear_regression";
import { LogisticRegression } from "./Regression/logistic_regression";

// Classification
const X_class = [
  [22, 20000],
  [25, 22000],
  [47, 48000],
  [52, 50000],
  [46, 45000],
  [56, 60000],
  [55, 62000],
  [60, 70000],
  [28, 25000],
  [30, 27000],
];

const y_class = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0];

const Knn = new KNearestNeighours(5);
Knn.fit(X_class, y_class);
console.log("KNN Prediction:", Knn.predict([[29, 24000]]));

const logRegress = new LogisticRegression(0.001, 100);
logRegress.fit(X_class, y_class);
console.log(
  "Logistic Regression Prediction:",
  logRegress.predict([[29, 24000]])
);

const DTclassifier = new DecisionTreeClassifier(2);
let tree = DTclassifier.fit(X_class, y_class);
console.log(
  `Decsion Tree prediction ${DTclassifier.predict(tree, [[29, 24000]])}`
);

const RFclassifier = new RandomForestClassifier(2);
let forest = RFclassifier.fit(X_class, y_class);
console.log(
  `Random Forest prediction ${DTclassifier.predict(tree, [[29, 24000]])}`
);

//Regression
const X = [[1], [2], [3], [4], [6], [7], [8], [10], [11]];
const y = [8, 11, 14, 17, 23, 26, 29, 35, 38];

const linRegres = new LinearRegression(0.0001, 1000);
linRegres.fit(X, y);
console.log(
  `Linear Regression Prediction ${linRegres.get_output([[5], [9], [12]])}`
);

const DTreg = new DecisionTreeRegressor();
const tree1 = DTreg.fit(X, y);
console.log(
  `Decision Tree Regressor ${DTreg.predict(tree1, [[5], [9], [12]])}`
);

const RFReg = new RandomForestRegressor();
const forest1 = RFReg.fit(X, y);
console.log(
  `Random Forest Regressor ${RFReg.predict(forest1, [[5], [9], [12]])}`
);
