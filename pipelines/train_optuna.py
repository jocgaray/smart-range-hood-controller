import argparse
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import dvc.api
import joblib
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*eval_set.*")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Reproducibility / provenance
# ---------------------------------------------------------------------------


def get_git_commit() -> str:
    """Return the current Git commit after checking for uncommitted changes."""
    try:
        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
            )
            .decode("utf-8")
            .strip()
        )

        if status and os.getenv("SKIP_GIT_CHECK") != "1":
            raise RuntimeError(
                "Uncommitted changes detected. Commit or stash changes before training."
            )

        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
            )
            .decode("utf-8")
            .strip()
        )

    except subprocess.CalledProcessError:
        return "unknown"


def get_dvc_data_version(data_path: str) -> str:
    """Retrieve the DVC-tracked dataset reference relative to project root."""
    try:
        path_obj = Path(data_path)
        rel_path = (
            path_obj.relative_to(PROJECT_ROOT) if path_obj.is_absolute() else path_obj
        )
        return dvc.api.get_url(
            path=str(rel_path),
            repo=str(PROJECT_ROOT),
        )
    except Exception:  # noqa: BLE001
        return "local-untracked"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


def suggest_model_params(trial: optuna.Trial) -> dict[str, Any]:
    """Return LightGBM hyperparameter search configuration."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_and_optimize(
    feature_data_path: str,
    output_model_path: str,
    n_trials: int = 30,
) -> None:
    """Train, evaluate, and track model with full metric logging."""
    start_time = time.time()

    mlflow.set_experiment("example_model_training")
    mlflow.lightgbm.autolog(log_models=False)

    with mlflow.start_run(run_name="example_optimization"):
        # -------------------------------------------------------------------
        # Provenance
        # -------------------------------------------------------------------

        git_hash = get_git_commit()
        dvc_data_reference = get_dvc_data_version(feature_data_path)

        mlflow.set_tag("git_commit", git_hash)
        mlflow.log_param("data_path", feature_data_path)
        mlflow.log_param("dvc_data_reference", dvc_data_reference)
        mlflow.log_param("n_trials", n_trials)

        # -------------------------------------------------------------------
        # Load data
        # -------------------------------------------------------------------

        print("[1/4] Loading feature data...")

        df = pd.read_parquet(feature_data_path)

        target_col = "target"
        if target_col not in df.columns:
            if "is_cooking" in df.columns:
                target_col = "is_cooking"
            elif "household_activity" in df.columns:
                target_col = "household_activity"

        group_col = "date_group"
        non_feature_cols = ["timestamp", group_col, target_col]
        candidate_cols = [
            column for column in df.columns if column not in non_feature_cols
        ]

        X_df = df[candidate_cols].select_dtypes(include=["number", "bool"])
        feature_cols = X_df.columns.tolist()

        X = X_df.to_numpy(dtype=np.float32)
        y = df[target_col].to_numpy(dtype=np.int32)
        groups = df[group_col].to_numpy()

        mlflow.log_param("active_feature_count", X.shape[1])
        mlflow.log_param("sample_count", len(X))

        print(f"  -> Features: {X.shape[1]} | Samples: {len(X):,}")

        # -------------------------------------------------------------------
        # Hyperparameter optimization
        # -------------------------------------------------------------------

        print("\n[2/4] Starting hyperparameter optimization...")

        progress = tqdm(
            total=n_trials,
            desc="Optimization trials",
            unit="trial",
            dynamic_ncols=True,
        )

        def objective(trial: optuna.Trial) -> float:
            trial_start = time.time()
            params = suggest_model_params(trial)

            with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                mlflow.log_params(params)

                unique_groups = len(np.unique(groups))
                if unique_groups >= 2:
                    n_splits = min(3, unique_groups)
                    cv = GroupKFold(n_splits=n_splits)
                    splits = list(cv.split(X, y, groups=groups))
                else:
                    split_idx = int(len(X) * 0.8)
                    splits = [(np.arange(0, split_idx), np.arange(split_idx, len(X)))]

                fold_scores: list[float] = []

                for _fold, (train_idx, val_idx) in enumerate(splits):
                    X_train, y_train = X[train_idx], y[train_idx]
                    X_val, y_val = X[val_idx], y[val_idx]

                    model = lgb.LGBMClassifier(**params)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
                    )

                    val_probs = np.asarray(model.predict_proba(X_val))[:, 1]
                    precision, recall, _ = precision_recall_curve(y_val, val_probs)
                    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
                    fold_scores.append(float(np.max(f1_scores)))

                score = float(np.mean(fold_scores))

                mlflow.log_metric("validation_score", score)
                mlflow.log_metric("trial_duration_seconds", time.time() - trial_start)

                progress.update(1)
                return score

        study = optuna.create_study(direction="maximize")

        try:
            study.optimize(
                objective,
                n_trials=n_trials,
                n_jobs=1,
                show_progress_bar=False,
            )
        finally:
            progress.close()

        print("\nOptimization complete.")

        # -------------------------------------------------------------------
        # Final evaluation & threshold optimization
        # -------------------------------------------------------------------

        print("[3/4] Evaluating selected model configuration & threshold...")

        best_params: dict[str, Any] = dict(study.best_params)
        best_params.update({"random_state": 42, "n_jobs": -1, "verbose": -1})

        mlflow.log_params({f"selected_{key}": value for key, value in best_params.items()})

        unique_groups = len(np.unique(groups))
        if unique_groups >= 2:
            n_splits = min(3, unique_groups)
            cv = GroupKFold(n_splits=n_splits)
            splits = list(cv.split(X, y, groups=groups))
        else:
            split_idx = int(len(X) * 0.8)
            splits = [(np.arange(0, split_idx), np.arange(split_idx, len(X)))]

        oof_preds = np.zeros(len(X), dtype=np.float32)
        fold_scores: list[float] = []

        for train_idx, val_idx in splits:
            model = lgb.LGBMClassifier(**best_params)
            model.fit(
                X[train_idx],
                y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)],
            )

            val_probs = np.asarray(model.predict_proba(X[val_idx]))[:, 1]
            oof_preds[val_idx] = val_probs

            precision, recall, _ = precision_recall_curve(y[val_idx], val_probs)
            f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
            fold_scores.append(float(np.max(f1_scores)))

        # Global Out-Of-Fold (OOF) Evaluation
        precisions, recalls, thresholds = precision_recall_curve(y, oof_preds)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)

        opt_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        best_f1_score = float(f1_scores[best_idx])

        oof_binary = (oof_preds >= opt_threshold).astype(int)
        opt_precision = float(precision_score(y, oof_binary, zero_division=0))
        opt_recall = float(recall_score(y, oof_binary, zero_division=0))
        roc_auc_val = float(roc_auc_score(y, oof_preds))
        pr_auc_val = float(auc(recalls, precisions))

        # Log complete metrics to top-level MLflow run
        mlflow.log_metric("best_validation_score", float(study.best_value))
        mlflow.log_metric("evaluation_mean_f1", float(np.mean(fold_scores)))
        mlflow.log_metric("oof_peak_f1", best_f1_score)
        mlflow.log_metric("optimal_threshold", opt_threshold)
        mlflow.log_metric("precision_at_optimal_threshold", opt_precision)
        mlflow.log_metric("recall_at_optimal_threshold", opt_recall)
        mlflow.log_metric("roc_auc", roc_auc_val)
        mlflow.log_metric("pr_auc", pr_auc_val)

        print(f"  -> Optimal Threshold: {opt_threshold:.4f}")
        print(f"  -> OOF Precision:     {opt_precision:.4f}")
        print(f"  -> OOF Recall:        {opt_recall:.4f}")
        print(f"  -> OOF Max F1:        {best_f1_score:.4f}")
        print(f"  -> ROC-AUC:           {roc_auc_val:.4f}")
        print(f"  -> PR-AUC:            {pr_auc_val:.4f}")

        # -------------------------------------------------------------------
        # Final model
        # -------------------------------------------------------------------

        print("[4/4] Training final model and exporting artifact...")

        final_model = lgb.LGBMClassifier(**best_params)
        final_model.fit(X, y)

        output_file = Path(output_model_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        pipeline_artifact = {
            "model": final_model,
            "feature_cols": feature_cols,
            "best_params": best_params,
            "optimal_threshold": opt_threshold,
        }

        joblib.dump(pipeline_artifact, output_file)

        mlflow.log_artifact(output_model_path, artifact_path="model_artifacts")

        mlflow.lightgbm.log_model(
            lgb_model=final_model,
            artifact_path="model",
            registered_model_name="example_classifier",
        )

        mlflow.log_metric("training_duration_seconds", time.time() - start_time)

        print("Training complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example ML training pipeline.")

    parser.add_argument(
        "--data",
        type=str,
        default="data/features_prepared.parquet",
        help="Path to engineered features parquet file",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="models/event_detector.joblib",
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=30,
    )

    args = parser.parse_args()

    train_and_optimize(
        args.data,
        args.output,
        n_trials=args.trials,
    )

# import argparse
# import subprocess
# import sys
# import time
# import warnings
# from pathlib import Path
# from typing import Any

# import dvc.api
# import joblib
# import lightgbm as lgb
# import mlflow
# import mlflow.lightgbm
# import numpy as np
# import optuna
# import pandas as pd
# from sklearn.metrics import precision_recall_curve
# from sklearn.model_selection import GroupKFold
# from tqdm import tqdm

# # ---------------------------------------------------------------------------
# # Project setup
# # ---------------------------------------------------------------------------

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.append(str(PROJECT_ROOT))

# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore", message=".*eval_set.*")
# optuna.logging.set_verbosity(optuna.logging.WARNING)


# # ---------------------------------------------------------------------------
# # Reproducibility / provenance
# # ---------------------------------------------------------------------------

# import os
# import subprocess


# def get_git_commit() -> str:
#     """Return the current Git commit after checking for uncommitted changes.

#     Production-specific reproducibility configuration is intentionally
#     simplified for this public example.
#     """
#     try:
#         status = (
#             subprocess.check_output(
#                 ["git", "status", "--porcelain"],
#                 cwd=PROJECT_ROOT,
#             )
#             .decode("utf-8")
#             .strip()
#         )

#         if status and os.getenv("SKIP_GIT_CHECK") != "1":
#             raise RuntimeError(
#                 "Uncommitted changes detected. Commit or stash changes before training."
#             )

#         return (
#             subprocess.check_output(
#                 ["git", "rev-parse", "HEAD"],
#                 cwd=PROJECT_ROOT,
#             )
#             .decode("utf-8")
#             .strip()
#         )

#     except subprocess.CalledProcessError:
#         return "unknown"

# # def get_git_commit() -> str:
# #     """Return the current Git commit after checking for uncommitted changes.

# #     Production-specific reproducibility configuration is intentionally
# #     simplified for this public example.
# #     """
# #     try:
# #         status = (
# #             subprocess.check_output(
# #                 ["git", "status", "--porcelain"],
# #                 cwd=PROJECT_ROOT,
# #             )
# #             .decode("utf-8")
# #             .strip()
# #         )

# #         if status:
# #             raise RuntimeError(
# #                 "Uncommitted changes detected. Commit or stash changes before training."
# #             )

# #         return (
# #             subprocess.check_output(
# #                 ["git", "rev-parse", "HEAD"],
# #                 cwd=PROJECT_ROOT,
# #             )
# #             .decode("utf-8")
# #             .strip()
# #         )

# #     except subprocess.CalledProcessError:
# #         return "unknown"


# def get_dvc_data_version(data_path: str) -> str:
#     """Retrieve the DVC-tracked dataset reference relative to project root."""
#     try:
#         path_obj = Path(data_path)
#         rel_path = (
#             path_obj.relative_to(PROJECT_ROOT) if path_obj.is_absolute() else path_obj
#         )
#         return dvc.api.get_url(
#             path=str(rel_path),
#             repo=str(PROJECT_ROOT),
#         )
#     except Exception:  # noqa: BLE001
#         return "local-untracked"


# # ---------------------------------------------------------------------------
# # Model configuration
# # ---------------------------------------------------------------------------


# def suggest_model_params(trial: optuna.Trial) -> dict[str, Any]:
#     """Return a sanitized LightGBM search configuration.

#     The production hyperparameter search space is intentionally not exposed.
#     """
#     return {
#         "n_estimators": trial.suggest_int(
#             "n_estimators",
#             100,
#             300,
#         ),
#         "learning_rate": trial.suggest_float(
#             "learning_rate",
#             0.01,
#             0.1,
#             log=True,
#         ),
#         # Additional production search parameters are intentionally omitted.
#         "random_state": 42,
#         "n_jobs": -1,
#         "verbose": -1,
#     }


# # ---------------------------------------------------------------------------
# # Training
# # ---------------------------------------------------------------------------


# def train_and_optimize(
#     feature_data_path: str,
#     output_model_path: str,
#     n_trials: int = 30,
# ) -> None:
#     """Train and evaluate a sanitized example classification pipeline.

#     The public implementation demonstrates:

#     - DVC data provenance
#     - Git-based reproducibility
#     - Optuna hyperparameter optimization
#     - LightGBM model training
#     - group-aware validation
#     - MLflow experiment tracking
#     - model artifact logging

#     Production feature engineering, model configuration, evaluation
#     methodology, and decision logic are intentionally omitted.
#     """
#     start_time = time.time()

#     # -----------------------------------------------------------------------
#     # MLflow experiment
#     # -----------------------------------------------------------------------

#     mlflow.set_experiment("example_model_training")

#     # Avoid excessive model logging during hyperparameter search.
#     mlflow.lightgbm.autolog(log_models=False)

#     with mlflow.start_run(run_name="example_optimization"):
#         # -------------------------------------------------------------------
#         # Provenance
#         # -------------------------------------------------------------------

#         git_hash = get_git_commit()
#         dvc_data_reference = get_dvc_data_version(feature_data_path)

#         mlflow.set_tag("git_commit", git_hash)
#         mlflow.log_param("data_path", feature_data_path)
#         mlflow.log_param(
#             "dvc_data_reference",
#             dvc_data_reference,
#         )
#         mlflow.log_param("n_trials", n_trials)

#         # -------------------------------------------------------------------
#         # Load data
#         # -------------------------------------------------------------------

#         print("[1/4] Loading feature data...")

#         df = pd.read_parquet(feature_data_path)

#         # Public example uses generic column names with dynamic fallback resolution.
#         target_col = "target"
#         if target_col not in df.columns:
#             if "is_cooking" in df.columns:
#                 target_col = "is_cooking"
#             elif "household_activity" in df.columns:
#                 target_col = "household_activity"

#         group_col = "date_group"

#         non_feature_cols = [
#             "timestamp",
#             group_col,
#             target_col,
#         ]

#         candidate_cols = [
#             column for column in df.columns if column not in non_feature_cols
#         ]

#         X_df = df[candidate_cols].select_dtypes(include=["number", "bool"])

#         feature_cols = X_df.columns.tolist()

#         X = X_df.to_numpy(dtype=np.float32)
#         y = df[target_col].to_numpy(dtype=np.int32)
#         groups = df[group_col].to_numpy()

#         # Only aggregate dataset characteristics are logged.
#         mlflow.log_param(
#             "active_feature_count",
#             X.shape[1],
#         )
#         mlflow.log_param(
#             "sample_count",
#             len(X),
#         )

#         print(f"  -> Features: {X.shape[1]} | Samples: {len(X):,}")

#         # -------------------------------------------------------------------
#         # Hyperparameter optimization
#         # -------------------------------------------------------------------

#         print("\n[2/4] Starting hyperparameter optimization...")

#         progress = tqdm(
#             total=n_trials,
#             desc="Optimization trials",
#             unit="trial",
#             dynamic_ncols=True,
#         )

#         def objective(trial: optuna.Trial) -> float:
#             trial_start = time.time()
#             params = suggest_model_params(trial)

#             with mlflow.start_run(
#                 run_name=f"trial_{trial.number}",
#                 nested=True,
#             ):
#                 mlflow.log_params(params)

#                 # Pure group-aware validation; dynamically adjusts folds to unique group count
#                 unique_groups = len(np.unique(groups))
#                 if unique_groups >= 2:
#                     n_splits = min(3, unique_groups)
#                     cv = GroupKFold(n_splits=n_splits)
#                     splits = list(cv.split(X, y, groups=groups))
#                 else:
#                     # Single-group fallback: 80/20 train/val split to prevent cross-group leakage
#                     split_idx = int(len(X) * 0.8)
#                     splits = [(np.arange(0, split_idx), np.arange(split_idx, len(X)))]

#                 fold_scores: list[float] = []

#                 for _fold, (train_idx, val_idx) in enumerate(splits):
#                     X_train = X[train_idx]
#                     y_train = y[train_idx]

#                     X_val = X[val_idx]
#                     y_val = y[val_idx]

#                     model = lgb.LGBMClassifier(**params)

#                     model.fit(
#                         X_train,
#                         y_train,
#                         eval_set=[(X_val, y_val)],
#                         callbacks=[
#                             lgb.early_stopping(
#                                 stopping_rounds=15,
#                                 verbose=False,
#                             )
#                         ],
#                     )

#                     val_probs = np.asarray(model.predict_proba(X_val))[:, 1]

#                     precision, recall, _ = precision_recall_curve(
#                         y_val,
#                         val_probs,
#                     )

#                     f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

#                     fold_scores.append(float(np.max(f1_scores)))

#                 score = float(np.mean(fold_scores))

#                 mlflow.log_metric(
#                     "validation_score",
#                     score,
#                 )

#                 mlflow.log_metric(
#                     "trial_duration_seconds",
#                     time.time() - trial_start,
#                 )

#                 progress.update(1)

#                 return score

#         study = optuna.create_study(
#             direction="maximize",
#         )

#         try:
#             study.optimize(
#                 objective,
#                 n_trials=n_trials,
#                 n_jobs=1,
#                 show_progress_bar=False,
#             )
#         finally:
#             progress.close()

#         print("\nOptimization complete.")

#         # -------------------------------------------------------------------
#         # Final evaluation
#         # -------------------------------------------------------------------

#         print("[3/4] Evaluating selected model configuration...")

#         best_params: dict[str, Any] = dict(study.best_params)

#         best_params.update(
#             {
#                 "random_state": 42,
#                 "n_jobs": -1,
#                 "verbose": -1,
#             }
#         )

#         mlflow.log_params(
#             {f"selected_{key}": value for key, value in best_params.items()}
#         )

#         mlflow.log_metric(
#             "optimization_score",
#             float(study.best_value),
#         )

#         # Pure group-aware validation evaluation
#         unique_groups = len(np.unique(groups))
#         if unique_groups >= 2:
#             n_splits = min(3, unique_groups)
#             cv = GroupKFold(n_splits=n_splits)
#             splits = list(cv.split(X, y, groups=groups))
#         else:
#             split_idx = int(len(X) * 0.8)
#             splits = [(np.arange(0, split_idx), np.arange(split_idx, len(X)))]

#         scores: list[float] = []

#         for train_idx, val_idx in splits:
#             model = lgb.LGBMClassifier(**best_params)

#             model.fit(
#                 X[train_idx],
#                 y[train_idx],
#                 eval_set=[(X[val_idx], y[val_idx])],
#                 callbacks=[
#                     lgb.early_stopping(
#                         stopping_rounds=15,
#                         verbose=False,
#                     )
#                 ],
#             )

#             val_probs = np.asarray(model.predict_proba(X[val_idx]))[:, 1]

#             precision, recall, _ = precision_recall_curve(
#                 y[val_idx],
#                 val_probs,
#             )

#             f1_scores = 2 * precision * recall / (precision + recall + 1e-10)

#             scores.append(float(np.max(f1_scores)))

#         mean_score = float(np.mean(scores))

#         mlflow.log_metric(
#             "evaluation_score",
#             mean_score,
#         )

#         # -------------------------------------------------------------------
#         # Final model
#         # -------------------------------------------------------------------

#         print("[4/4] Training final model and exporting artifact...")

#         final_model = lgb.LGBMClassifier(**best_params)
#         final_model.fit(X, y)

#         output_file = Path(output_model_path)
#         output_file.parent.mkdir(
#             parents=True,
#             exist_ok=True,
#         )

#         # Only the generic model object and public metadata are represented
#         # here. Production feature engineering, thresholds, and model
#         # configuration are intentionally excluded.
#         pipeline_artifact = {
#             "model": final_model,
#             "feature_cols": feature_cols,
#         }

#         joblib.dump(
#             pipeline_artifact,
#             output_file,
#         )

#         mlflow.log_artifact(
#             output_model_path,
#             artifact_path="model_artifacts",
#         )

#         mlflow.lightgbm.log_model(
#             lgb_model=final_model,
#             artifact_path="model",
#             registered_model_name="example_classifier",
#         )

#         mlflow.log_metric(
#             "training_duration_seconds",
#             time.time() - start_time,
#         )

#         print("Training complete.")


# # ---------------------------------------------------------------------------
# # CLI
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Example ML training pipeline.")

#     parser.add_argument(
#         "--data",
#         type=str,
#         default="data/features_prepared.parquet",
#         help="Path to engineered features parquet file",
#     )

#     parser.add_argument(
#         "--output",
#         type=str,
#         default="models/event_detector.joblib",
#     )

#     parser.add_argument(
#         "--trials",
#         type=int,
#         default=30,
#     )

#     args = parser.parse_args()

#     train_and_optimize(
#         args.data,
#         args.output,
#         n_trials=args.trials,
#     )