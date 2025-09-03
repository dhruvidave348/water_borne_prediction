=import os
if X_valid is not None and y_valid is not None:
self.model.fit(
X_train, y_train,
early_stopping_rounds=20,
eval_set=[(X_valid, y_valid)],
verbose=False
)
else:
self.model.fit(X_train, y_train)


def predict(self, X):
return self.model.predict(X)


def predict_proba(self, X):
return self.model.predict_proba(X)[:, 1]


def evaluate(self, X_test, y_test):
preds = self.predict(X_test)
probs = self.predict_proba(X_test)
acc = accuracy_score(y_test, preds)
roc = roc_auc_score(y_test, probs)
cm = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds)
return dict(accuracy=acc, roc_auc=roc, confusion_matrix=cm, report=report)


def feature_importances(self, feature_names):
# returns pandas Series-like (list) sorted by importance
importances = self.model.feature_importances_
return list(zip(feature_names, importances))


def save(self, path=None):
path = path or self.model_path
if not path:
raise ValueError('No path provided to save the model')
os.makedirs(os.path.dirname(path), exist_ok=True)
joblib.dump(self.model, path)


def load(self, path):
self.model = joblib.load(path)
self.model_path = path