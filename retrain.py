import datetime
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

BASE_DIR = "advisory_app"
df = pd.read_csv(f"{BASE_DIR}/cleaned_indian_startups.csv")

print("Dataset shape:", df.shape)
print("Class distribution:\n", df["investment_recommendation"].value_counts())

# ── Target ────────────────────────────────────────────────────────────
y = df["investment_recommendation"]  # three classes: Invest / Hold / Avoid

# ── Features ──────────────────────────────────────────────────────────
DROP = ["startup_name", "founder_names", "investment_score_0_100",
        "investment_recommendation", "data_verified"]
X_raw = df.drop(columns=DROP)

# Full dummies (no drop_first — avoids baseline ambiguity for imputation)
X = pd.get_dummies(X_raw)
all_features = X.columns.tolist()

# ── Train / Test split (stratified) ───────────────────────────────────
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {len(X_train_df)} | Test size: {len(X_test_df)}")
print("Train class distribution:\n", y_train.value_counts())

# ── Imputation defaults (training split only) ─────────────────────────
OPTIONAL_NUMERIC = [
    "num_founders", "prev_startups", "tech_founder",
    "employees", "funding_rounds", "seed_usd_mn",
    "series_a_usd_mn", "series_b_usd_mn", "burn_rate_usd_mn_monthly",
    "competitor_count", "website_traffic_mn_monthly", "app_downloads_mn",
    "social_media_followers_mn", "year_founded", "company_age",
    "last_funding_year", "time_between_rounds_months", "market_growth_rate_pct",
    "is_unicorn", "valuation_usd_mn",
]
OPTIONAL_CATEGORICAL = ["industry", "city", "education_level", "ipo_status"]

X_raw_train = X_raw.loc[X_train_df.index]

imputation_defaults = {"numeric": {}, "categorical": {}}
for col in OPTIONAL_NUMERIC:
    if col in X_raw_train.columns:
        imputation_defaults["numeric"][col] = float(X_raw_train[col].mean())
for col in OPTIONAL_CATEGORICAL:
    if col in X_raw_train.columns:
        imputation_defaults["categorical"][col] = str(X_raw_train[col].mode()[0])

print("\nImputation defaults (categorical):", imputation_defaults["categorical"])

# ── Label encode target ───────────────────────────────────────────────
le = LabelEncoder()
le.fit(y)                              # fit on all so ordering is stable
y_train_enc = le.transform(y_train)   # Avoid=0, Hold=1, Invest=2
y_test_enc  = le.transform(y_test)
print("\nLabel encoding:", dict(zip(le.classes_, le.transform(le.classes_))))

# ── SMOTE on training set only ────────────────────────────────────────
# k_neighbors=2 because Invest class has only ~12 training samples
sm = SMOTE(random_state=42, k_neighbors=2)
X_train_res, y_train_res = sm.fit_resample(X_train_df, y_train_enc)
X_train_res = pd.DataFrame(X_train_res, columns=X_train_df.columns)
print("\nAfter SMOTE class distribution:", dict(zip(*np.unique(y_train_res, return_counts=True))))

# ── Variance threshold feature selection ──────────────────────────────
selector = VarianceThreshold(threshold=0.01)
X_train_sel = selector.fit_transform(X_train_res)
X_test_sel  = selector.transform(X_test_df)

selected_features = X_train_df.columns[selector.get_support()]
print(f"\nFeatures after selection: {len(selected_features)} / {len(all_features)}")

X_train_sel_df = pd.DataFrame(X_train_sel, columns=selected_features)
X_test_sel_df  = pd.DataFrame(X_test_sel,  columns=selected_features)

# ── Train ──────────────────────────────────────────────────────────────
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_child_samples=10,
    random_state=42,
    verbose=-1,
)
model.fit(X_train_sel_df, y_train_res)
print("\nModel trained.")

# ── Evaluate on held-out test set ─────────────────────────────────────
y_pred = model.predict(X_test_sel_df)
acc    = accuracy_score(y_test_enc, y_pred)
report = classification_report(
    y_test_enc, y_pred,
    target_names=le.classes_,
    output_dict=True,
    zero_division=0,
)

print(f"\nTest Accuracy : {acc:.4f}")
print(f"Weighted F1   : {report['weighted avg']['f1-score']:.4f}")
print("\nPer-class results:")
for cls in le.classes_:
    r = report[cls]
    print(f"  {cls:8s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1-score']:.3f}  n={int(r['support'])}")

# ── 5-fold stratified cross-validation on training set ────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = cross_val_score(
    model, X_train_sel_df, y_train_res,
    cv=cv,
    scoring="f1_weighted",
)
print(f"\nCV F1 (5-fold): {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")

# ── Build metrics JSON ─────────────────────────────────────────────────
metrics = {
    "test_accuracy":  round(acc, 4),
    "weighted_f1":    round(report["weighted avg"]["f1-score"], 4),
    "cv_f1_mean":     round(float(cv_f1_scores.mean()), 4),
    "cv_f1_std":      round(float(cv_f1_scores.std()), 4),
    "train_test_split": "80/20",
    "train_size":     len(X_train_df),
    "test_size":      len(X_test_df),
    "smote":          "applied to training set only (k_neighbors=2)",
    "generated_at":   datetime.datetime.now().isoformat(),
    "per_class":      {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall":    round(report[cls]["recall"], 4),
            "f1":        round(report[cls]["f1-score"], 4),
            "support":   int(report[cls]["support"]),
        }
        for cls in le.classes_
    },
}

# ── Save all artefacts ─────────────────────────────────────────────────
joblib.dump(model,             f"{BASE_DIR}/startup_model.pkl")
joblib.dump(selector,          f"{BASE_DIR}/feature_selector.pkl")
joblib.dump(selected_features, f"{BASE_DIR}/feature_names.pkl")
joblib.dump(all_features,      f"{BASE_DIR}/all_features.pkl")
joblib.dump(imputation_defaults, f"{BASE_DIR}/imputation_defaults.pkl")
joblib.dump(le,                f"{BASE_DIR}/label_encoder.pkl")

with open(f"{BASE_DIR}/model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\nAll artefacts saved:")
for name in ["startup_model.pkl", "feature_selector.pkl", "feature_names.pkl",
             "all_features.pkl", "imputation_defaults.pkl", "label_encoder.pkl",
             "model_metrics.json"]:
    print(f"  advisory_app/{name}")
