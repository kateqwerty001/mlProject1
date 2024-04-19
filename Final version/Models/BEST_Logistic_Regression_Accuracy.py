import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from basic_pipeline_functions import PipelineBasic
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data_all = pd.read_csv('../data/data.csv')

train, test = train_test_split(data_all, test_size=0.2, random_state=42)

X_train = train.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_train = train['DEFAULT']

X_test = test.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_test = test['DEFAULT']

log_clf = LogisticRegression(random_state=42, max_iter=300, C=0.01, penalty='l2', solver='liblinear')
selector = RFE(estimator=log_clf, n_features_to_select=17, step=1)

LR_pipeline = Pipeline([
    ('basic_pipeline', PipelineBasic),
    ('scaler', StandardScaler()),
    ('feature_selection', selector),
    ('classifier', log_clf)
])

print("______________CROSS VALIDATION_________________________________________________________")
y_pred_cv = cross_val_predict(LR_pipeline, X_train, y_train, cv=5)

precision_0_cv = precision_score(y_train, y_pred_cv, pos_label=0)
recall_0_cv = recall_score(y_train, y_pred_cv, pos_label=0)
precision_1_cv = precision_score(y_train, y_pred_cv, pos_label=1)
recall_1_cv = recall_score(y_train, y_pred_cv, pos_label=1)

print("Precision for class 0 (cross-validation):", precision_0_cv)
print("Recall for class 0 (cross-validation):", recall_0_cv)
print("Precision for class 1 (cross-validation):", precision_1_cv)
print("Recall for class 1 (cross-validation):", recall_1_cv)

conf_matrix_cv = confusion_matrix(y_train, y_pred_cv)

print("Confusion Matrix (cross-validation):")
print(conf_matrix_cv)

print("Accuracy (cross-validation):", (conf_matrix_cv[0][0] + conf_matrix_cv[1][1]) / (conf_matrix_cv[0][0] + conf_matrix_cv[0][1] + conf_matrix_cv[1][0] + conf_matrix_cv[1][1]))

print("______________TESTING_________________________________________________________")

LR_pipeline.fit(X_train, y_train)
y_pred = LR_pipeline.predict(X_test)

#accuracy matrix

precision_0 = precision_score(y_test, y_pred, pos_label=0)
recall_0 = recall_score(y_test, y_pred, pos_label=0)
precision_1 = precision_score(y_test, y_pred, pos_label=1)
recall_1 = recall_score(y_test, y_pred, pos_label=1)

print("Precision for class 0:", precision_0)
print("Recall for class 0:", recall_0)
print("Precision for class 1:", precision_1)
print("Recall for class 1:", recall_1)

conf_matrix = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)

print("Accuracy:", (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1]))

"""
FOCUSED ON BEST ACCURACY
______________CROSS VALIDATION FOR THE BEST PARAMETERS FROM GRID SEARCH_________________________________________________________
Precision for class 0 (cross-validation): 0.7545126353790613
Recall for class 0 (cross-validation): 0.9106753812636166
Precision for class 1 (cross-validation): 0.5232558139534884
Recall for class 1 (cross-validation): 0.24861878453038674
Confusion Matrix (cross-validation):
[[418  41]
 [136  45]]
Accuracy (cross-validation): 0.7234375
______________TESTING__________________________________________________________________________________________________
Precision for class 0: 0.7761194029850746
Recall for class 0: 0.9122807017543859
Precision for class 1: 0.6153846153846154
Recall for class 1: 0.34782608695652173
Confusion Matrix:
[[104  10]
 [ 30  16]]
Accuracy: 0.75
"""

