import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from basic_pipeline_functions import PipelineBasic
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score

data_all = pd.read_csv('../data/data.csv')

train, test = train_test_split(data_all, test_size=0.2, random_state=42)

X_train = train.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_train = train['DEFAULT']

X_test = test.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_test = test['DEFAULT']

clf = GradientBoostingClassifier(learning_rate=0.16035155241954244,
                                 max_depth=3,
                                 min_samples_leaf=0.09149844363892358,
                                 min_samples_split=0.6138774053515396,
                                 n_estimators=19,
                                 subsample=0.6859413384082682,
                                 random_state=42)

pca = PCA(n_components=23)

pipeline = Pipeline([
    ('basic_pipeline', PipelineBasic),
    ('pca', pca),
    ('classifier', clf)
])

print("______________CROSS VALIDATION_________________________________________________________")
y_pred_cv = cross_val_predict(pipeline, X_train, y_train, cv=10)

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

accuracy_cv = (conf_matrix_cv[0][0] + conf_matrix_cv[1][1]) / (conf_matrix_cv[0][0] + conf_matrix_cv[0][1] + conf_matrix_cv[1][0] + conf_matrix_cv[1][1])
print("Accuracy (cross-validation):", accuracy_cv)

print("______________TESTING_________________________________________________________")

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

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

accuracy = (conf_matrix[0][0] + conf_matrix[1][1]) / (conf_matrix[0][0] + conf_matrix[0][1] + conf_matrix[1][0] + conf_matrix[1][1])
print("Accuracy:", accuracy)

"""
______________CROSS VALIDATION_________________________________________________________
Precision for class 0 (cross-validation): 0.7336601307189542
Recall for class 0 (cross-validation): 0.9782135076252724
Precision for class 1 (cross-validation): 0.6428571428571429
Recall for class 1 (cross-validation): 0.09944751381215469
Confusion Matrix (cross-validation):
[[449  10]
 [163  18]]
Accuracy (cross-validation): 0.7296875
______________TESTING_________________________________________________________
Precision for class 0: 0.75
Recall for class 0: 1.0
Precision for class 1: 1.0
Recall for class 1: 0.17391304347826086
Confusion Matrix:
[[114   0]
 [ 38   8]]
Accuracy: 0.7625
"""