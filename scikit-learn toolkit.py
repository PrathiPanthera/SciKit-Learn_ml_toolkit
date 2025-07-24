# ✅ Regression Models

# 1. Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Ridge Regression
from sklearn.linear_model import Ridge
model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Lasso Regression
from sklearn.linear_model import Lasso
model = Lasso()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. ElasticNet Regression
from sklearn.linear_model import ElasticNet
model = ElasticNet()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. AdaBoost Regressor
from sklearn.ensemble import AdaBoostRegressor
model = AdaBoostRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9. Support Vector Regressor
from sklearn.svm import SVR
model = SVR()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 10. KNeighbors Regressor
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ✅ Classification Models

# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 2. Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 3. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 4. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. AdaBoost Classifier
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Support Vector Classifier
from sklearn.svm import SVC
model = SVC()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 8. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 9. Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 10. MLP Classifier (Neural Network)
from sklearn.neural_network import MLPClassifier
model = MLPClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🎯 Accuracy Evaluation Example (for classification)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 🔹 Voting & Stacking Classifiers
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Example usage - Voting Classifier (Hard Voting)
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GradientBoostingClassifier()
voting = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('gb', clf3)
], voting='hard')
voting.fit(X_train, y_train)
y_pred = voting.predict(X_test)

# Example usage - Stacking Classifier
base_learners = [
    ('rf', RandomForestClassifier()),
    ('gb', GradientBoostingClassifier())
]
meta_learner = LogisticRegression()
stacking = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)
stacking.fit(X_train, y_train)
y_pred = stacking.predict(X_test)

# 🔹 Bagging Regressor & Classifier
from sklearn.ensemble import BaggingClassifier, BaggingRegressor

# Example usage - BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bagging = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10)
bagging.fit(X_train, y_train)
y_pred = bagging.predict(X_test)

# Example usage - BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
bagreg = BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=10)
bagreg.fit(X_train, y_train)
y_pred = bagreg.predict(X_test)

# 🔹 Clustering Algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Example usage - KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_labels = kmeans.labels_

# Example usage - DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_labels = dbscan.fit_predict(X)

# Example usage - Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=3)
y_labels = agglo.fit_predict(X)

# 🔹 Datasets for quick testing
from sklearn.datasets import load_iris, load_diabetes
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# 🔹 Permutation Importance
from sklearn.inspection import permutation_importance

# Example usage:
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
importance = result.importances_mean
print("Permutation Feature Importance:", importance)
