from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    RidgeCV,
    Ridge,
)

from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline

# ### Dict of estimators for 1D toy examples ### #
CLASSIFIERS = {
    "linear": LogisticRegression(),
    "linear_tlearn": Pipeline(
        [
            ("interaction", PolynomialFeatures(interaction_only=True)),
            ("reg", LogisticRegression()),
        ]
    ),
    "linearCV": LogisticRegressionCV(),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=1),
    "mlp": MLPClassifier(hidden_layer_sizes=(100, 100, 100, 100), random_state=0),
    "svc_rbf": SVC(probability=True, random_state=0, C=0.2),
    # "svc_rbf": SVC(probability=True, random_state=0, C=10),
    "svm_lin": SVC(kernel="linear", probability=True, random_state=0),
    "svm_poly": SVC(kernel="poly", degree=2, probability=True, random_state=0),
    "random_forest": RandomForestClassifier(random_state=0, n_estimators=20),
    "poly": make_pipeline(PolynomialFeatures(), LogisticRegression()),
    "hist_gradient_boosting": HistGradientBoostingClassifier(),
}

REGRESSORS = {
    "linear": LinearRegression(),
    "ridge": Ridge(),
    "linear_tlearn": Pipeline(
        [
            ("interaction", PolynomialFeatures(interaction_only=True)),
            ("reg", LinearRegression()),
        ]
    ),
    "linearCV": RidgeCV(),
    "gradient_boosting": GradientBoostingRegressor(n_estimators=500, random_state=0),
    "svm_lin": SVR(kernel="linear"),
    "svm_poly": SVR(kernel="poly", degree=2),
    "random_forest": RandomForestRegressor(random_state=0, n_estimators=20),
    "poly3": make_pipeline(PolynomialFeatures(degree=3), Ridge()),
    "spline3": make_pipeline(SplineTransformer(degree=3), Ridge()),
    "hist_gradient_boosting": HistGradientBoostingRegressor(),
}

HP_KWARGS_HGB = {
    "histgradientboostingclassifier__learning_rate": [1e-3, 1e-2, 1e-1, 1],
    "histgradientboostingclassifier__min_samples_leaf": [2, 10, 50, 100, 200],
}

HP_KWARGS_LR = {"logisticregression__C": [1e-3, 1e-2, 1e-1, 1]}
