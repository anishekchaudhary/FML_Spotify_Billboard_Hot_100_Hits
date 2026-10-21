"""Model families and valid search spaces for the Spotify notebook workflow."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingClassifier, GradientBoostingClassifier,
    GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor


FEATURES = [
    "duration_ms", "explicit", "danceability", "energy", "key", "loudness",
    "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "time_signature", "track_genre",
]


def make_pipeline(model, classification, seed):
    # TargetEncoder.fit_transform cross-fits training encodings. Because this is
    # inside Pipeline, encoding, scaling and PCA are also fitted inside each CV fold.
    encoding = ColumnTransformer([
        ("genre", TargetEncoder(
            target_type="binary" if classification else "continuous",
            cv=5, shuffle=True, random_state=seed,
        ), ["track_genre"]),
        ("numeric", "passthrough", [c for c in FEATURES if c != "track_genre"]),
    ])
    return Pipeline([
        ("encoding", encoding),
        ("scaling", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("model", model),
    ])


def model_specs(classification, config):
    """Retain original valid families; remove invalid parameter combinations."""
    seed = config["seed"]
    smoke = config["profile"] == "smoke"
    trees = [30, 50] if smoke else [100, 200, 300, 400, 500]
    depths = [5, 10] if smoke else [10, 20, 30, None]
    if classification:
        specs = {
            "LogisticRegression": (LogisticRegression(
                solver="liblinear", max_iter=1000, class_weight="balanced", random_state=seed,
            ), {"C": np.logspace(-4, 4, 20)}),
            "DecisionTreeClassifier": (DecisionTreeClassifier(
                random_state=seed, class_weight="balanced",
            ), {"criterion": ["gini", "entropy"], "max_depth": depths,
                "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2", None]}),
            "RandomForestClassifier": (RandomForestClassifier(
                random_state=seed, class_weight="balanced_subsample", n_jobs=1,
            ), {"n_estimators": trees, "max_depth": depths,
                "min_samples_split": [2, 5, 10], "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2"]}),
            "SVC": (SVC(random_state=seed, class_weight="balanced", cache_size=256),
                {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf", "poly", "sigmoid"],
                 "gamma": ["scale", "auto"]}),
            "KNeighborsClassifier": (KNeighborsClassifier(n_jobs=1),
                {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"],
                 "metric": ["euclidean", "manhattan", "minkowski"]}),
            "GaussianNB": (GaussianNB(), {"var_smoothing": np.logspace(-9, -1, 10)}),
            "GradientBoostingClassifier": (GradientBoostingClassifier(random_state=seed),
                {"n_estimators": trees, "learning_rate": [0.001, 0.01, 0.1, 0.2],
                 "max_depth": [3, 4] if smoke else [3, 4, 5, 6],
                 "subsample": [0.6, 0.8, 1.0], "min_samples_split": [2, 5, 10],
                 "min_samples_leaf": [1, 2, 5]}),
            "XGBClassifier": (XGBClassifier(
                random_state=seed, n_jobs=1, tree_method="hist", eval_metric="logloss",
            ), {"n_estimators": trees, "learning_rate": [0.001, 0.01, 0.1, 0.2],
                "max_depth": [3, 4] if smoke else [3, 4, 5, 6],
                "subsample": [0.6, 0.8, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3], "reg_alpha": [0, 0.1, 0.5],
                "reg_lambda": [1, 1.5, 2.0]}),
            "MLPClassifier": (MLPClassifier(random_state=seed, max_iter=100 if smoke else 500),
                {"hidden_layer_sizes": [(30,)] if smoke else [(50,), (100,), (100, 50), (100, 100)],
                 "activation": ["tanh", "relu"], "solver": ["adam"],
                 "alpha": [0.0001, 0.001, 0.01]}),
            "BaggingClassifier": (BaggingClassifier(random_state=seed, n_jobs=1),
                {"n_estimators": [10, 20] if smoke else [10, 20, 30, 50, 100],
                 "max_samples": [0.5, 0.7, 1.0], "max_features": [0.5, 0.7, 1.0],
                 "bootstrap": [True, False], "bootstrap_features": [True, False]}),
        }
    else:
        alphas = np.logspace(-4, 0, 100)
        specs = {
            "Lasso": (Lasso(random_state=seed),
                {"alpha": alphas, "max_iter": [1000, 5000, 10000],
                 "tol": [1e-4, 1e-3, 1e-2], "selection": ["cyclic", "random"]}),
            "Ridge": (Ridge(random_state=seed),
                {"alpha": alphas, "max_iter": [1000, 5000, 10000], "tol": [1e-4, 1e-3, 1e-2]}),
            "ElasticNet": (ElasticNet(random_state=seed),
                {"alpha": alphas, "l1_ratio": np.linspace(0.05, 1, 20),
                 "max_iter": [1000, 5000, 10000], "tol": [1e-4, 1e-3, 1e-2]}),
            "RandomForestRegressor": (RandomForestRegressor(random_state=seed, n_jobs=1),
                {"n_estimators": trees, "max_depth": depths,
                 "min_samples_split": [2, 5, 10, 15, 20], "min_samples_leaf": [1, 2, 4, 6, 8],
                 "max_features": [1.0, "sqrt", "log2"]}),
            "GradientBoostingRegressor": (GradientBoostingRegressor(random_state=seed),
                {"n_estimators": trees, "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                 "max_depth": [3, 4] if smoke else [3, 4, 5, 6, 7],
                 "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "min_samples_split": [2, 5, 10],
                 "min_samples_leaf": [1, 2, 4], "max_features": [None, "sqrt", "log2"]}),
            "AdaBoostRegressor": (AdaBoostRegressor(random_state=seed),
                {"n_estimators": trees if smoke else [50, 100, 150, 200, 250],
                 "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                 "loss": ["linear", "square", "exponential"]}),
            "KNeighborsRegressor": (KNeighborsRegressor(n_jobs=1),
                {"n_neighbors": [3, 5, 7, 9, 11, 13, 15], "weights": ["uniform", "distance"],
                 "metric": ["euclidean", "manhattan", "minkowski"]}),
            "XGBRegressor": (XGBRegressor(
                random_state=seed, n_jobs=1, tree_method="hist", objective="reg:squarederror",
            ), {"n_estimators": trees, "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 4] if smoke else [3, 4, 5, 6, 7],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0], "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "gamma": [0, 0.1, 0.2, 0.3, 0.4], "reg_alpha": [0, 0.1, 0.5, 1.0],
                "reg_lambda": [1, 1.5, 2.0, 2.5]}),
        }
    names = config["classifier_models" if classification else "regressor_models"]
    return {name: specs[name] for name in names}
