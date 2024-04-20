import pandas as pd
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from skopt import BayesSearchCV
from skopt.space import Real, Categorical
from basic_pipeline_functions import PipelineBasic
from sklearn.decomposition import PCA

data_all = pd.read_csv('../data/data.csv')

train, test = train_test_split(data_all, test_size=0.2, random_state=42)

X_train = train.drop(['CREDIT_SCORE', 'DEFAULT'], axis=1)
y_train = train['DEFAULT']

X_test = test.drop(['CREDIT_SCORE', 'DEFAULT'], axis=1)
y_test = test['DEFAULT']

XGB_pipeline = Pipeline([
    ('basic_pipeline', PipelineBasic),
    ('pca', PCA()),
    ('classifier', xgb.XGBClassifier(random_state=42))
])

param_space = {
    'pca__n_components': [None, 4,5,6,7,8,9],
    'classifier__learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'classifier__n_estimators': Categorical([17]),
    'classifier__max_depth': Categorical([3]),
    'classifier__min_child_weight': Real(0.01, 1.0, 'uniform'),
    'classifier__subsample': Real(0.5, 1.0, 'uniform'),
    'classifier__colsample_bytree': Real(0.5, 1.0, 'uniform'),
    'classifier__reg_alpha': Real(1e-9, 1000, 'log-uniform'),
    'classifier__reg_lambda': Real(1e-9, 1000, 'log-uniform'),
    'classifier__gamma': Real(1e-9, 0.5, 'uniform')
}

bayes_search = BayesSearchCV(
    XGB_pipeline,
    param_space,
    cv=KFold(n_splits=5, shuffle=True, random_state=42),
    n_iter=100,
    scoring='accuracy',
    random_state=42,
    n_jobs=-1
)

bayes_search.fit(X_train, y_train)

print("Best parameters on cross-validation:", bayes_search.best_params_)
"""
Best parameters on cross-validation: OrderedDict([('classifier__colsample_bytree', 0.6660921447539443), ('classifier__gamma', 0.05000040397304358), ('classifier__learning_rate', 0.03282455192996737), ('classifier__max_depth', 3), ('classifier__min_child_weight', 0.2878701521673102), ('classifier__n_estimators', 17), ('classifier__reg_alpha', 0.03496544671788251), ('classifier__reg_lambda', 1e-09), ('classifier__subsample', 0.9401249312670386), ('pca__n_components', 8)])
"""