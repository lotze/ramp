import urllib2

import pandas as pd
from ramp.folds import BalancedFolds
import sklearn
from sklearn import decomposition

import ramp
from ramp.features import *
from ramp.metrics import PositiveRate, Recall

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# fetch and clean iris data from UCI
# data = pd.read_csv('iris.csv')
data = pd.read_csv(urllib2.urlopen(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"))
data = data.drop([149]) # bad line
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data.columns = columns

data['is_setosa'] = data['class'] == 'Iris-setosa'

# all features
features = [FillMissing(f, 0) for f in columns[:-1]]

# features, log transformed features, and interaction terms
expanded_features = (
    features +
    [Log(F(f) + 1) for f in features] +
    [
        F('sepal_width') ** 2,
        combo.Interactions(features),
    ]
)

reporters = [
    ramp.reporters.MetricReporter.factory(Recall(.4)),
    ramp.reporters.DualThresholdMetricReporter.factory(Recall(), PositiveRate())
]

# get a single training and test set
folds=BalancedFolds(5, FillMissing('is_setosa', 0), data)
folds.build_target()
folds.compute_folds()
one_fold = folds.folds[0]
train_index = one_fold[0]
test_index = one_fold[1]
prep_index = None
training_data = data.iloc[train_index]
test_data = data.iloc[test_index]

# Define a single model and feature set
model_def = ramp.model_definition.ModelDefinition(target='is_setosa', features=expanded_features, estimator=sklearn.ensemble.RandomForestClassifier(n_estimators=20))

x_train, y_train, fitted_model = ramp.modeling.fit_model(model_def, data, prep_index, train_index)
x_test, y_test = ramp.modeling.generate_test(model_def, test_data, fitted_model)
y_preds = fitted_model.fitted_estimator.predict(x_test)
result = ramp.result.Result(x_train, x_test, y_train, y_test, y_preds, model_def, fitted_model, data)

# instantiate, update, and display the reporters
instantiated_reporters = [rf() for rf in reporters]
[r.update(result) for r in instantiated_reporters]
print "Single model on single training/test:"
print [str(r) for r in instantiated_reporters]

# Now try that model with cross-validation via  a factory
outcomes = ramp.shortcuts.cv_factory(
    data=data,
    folds=BalancedFolds(5, FillMissing('is_setosa', 0), data), # small data, need to ensure we have positive and negative examples in each training

    target=['is_setosa'],

    reporter_factories=reporters,

    # Try out one algorithms
    estimator=[
        sklearn.ensemble.RandomForestClassifier(
            n_estimators=20)
        ],

    # and one feature set
    features=[
        expanded_features
    ]
)
print "5-fold cross-validated single-model results:"
print [str(r) for r in outcomes.values()[0]['reporters']]

# Define several models and feature sets to explore,
# run 5 fold cross-validation on each and print the results.
# We define 2 models and 4 feature sets, so this will be
# 4 * 2 = 8 models tested.

outcomes = ramp.shortcuts.cv_factory(
    data=data,
    folds=BalancedFolds(5, FillMissing('is_setosa', 0), data), # small data, need to ensure we have positive and negative examples in each training

    target=['is_setosa'],

    reporter_factories=reporters,

    # Try out two algorithms
    estimator=[
        sklearn.ensemble.RandomForestClassifier(
            n_estimators=20),
        sklearn.linear_model.LogisticRegression(),
        ],

    # and 4 feature sets
    features=[
        expanded_features,

        # Feature selection
        # [trained.FeatureSelector(
        #     expanded_features,
        #     # use random forest's importance to trim
        #     ramp.selectors.BinaryFeatureSelector(),
        #     target=AsFactor('class'), # target to use
        #     data=data,
        #     n_keep=5, # keep top 5 features
        #     )],

        # Reduce feature dimension (pointless on this dataset)
        [combo.DimensionReduction(expanded_features,
                            decomposer=decomposition.PCA(n_components=4))],

        # Normalized features
        [Normalize(f) for f in expanded_features],
    ]
)
print "5-fold cross-validated multiple models and feature sets results:"
for model_spec in outcomes:
    print "Results for " + model_spec[0]
    print [str(r) for r in outcomes[model_spec]['reporters']]
