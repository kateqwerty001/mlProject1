import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from basic_pipeline_functions import PipelineBasic

data_all = pd.read_csv('../data/data.csv')

train, test = train_test_split(data_all, test_size=0.2, random_state=42)

X_train = train.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_train = train['DEFAULT']

X_test = test.drop(['CREDIT_SCORE','DEFAULT'], axis=1)
y_test = test['DEFAULT']

log_clf = LogisticRegression(random_state=42, max_iter=300)
selector = RFE(estimator=log_clf, n_features_to_select=1, step=1)

LR_pipeline = Pipeline([
    ('basic_pipeline', PipelineBasic),
    ('scaler', StandardScaler()),
    ('feature_selection', selector),
    ('classifier', log_clf)
])

param_grid = {
    'feature_selection__n_features_to_select': [17],
    'classifier__penalty': ['l2'],
    'classifier__solver': ['liblinear'],
#    'classifier__solver': ['liblinear', 'saga', 'lbfgs', 'newton-cg'],
    'classifier__C': [0.01],
}

grid_search = GridSearchCV(LR_pipeline, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')

grid_search.fit(X_train, y_train)
print("Best parameters on cross-validation:", grid_search.best_params_)
print("Best accuracy on cross-validation:", grid_search.best_score_)

# 1 grid search
# Best parameters on cross-validation: {'classifier__C': 0.01, 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear', 'feature_selection__n_features_to_select': 20}


# 2 grid search
# Best parameters on cross-validation: {'classifier__C': 0.01, 'classifier__penalty': 'l2', 'classifier__solver': 'liblinear', 'feature_selection__n_features_to_select': 17}
