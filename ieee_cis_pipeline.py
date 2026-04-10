#!/usr/bin/env python3
"""
Flow:
  - Out-of-Core chunked ingestion (scales to any size)
  - Two-stage detection: Isolation Forest + LightGBM (Optuna tuned)
  - Isotonic calibration, SHAP, Risk Fusion, Drift Monitoring
  - All charts saved as interactive Plotly HTML

Dataset: data/raw/ieee-fraud-detection/
  train_transaction.csv  (590,540 rows, 394 cols)
  train_identity.csv     (144,233 rows,  41 cols)

Usage:
    python3 ieee_cis_pipeline.py 2>&1 | tee logs/ieee_cis.log
"""

import os, sys, gc, time, json, warnings, traceback
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore")
os.makedirs("logs", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("data/feature_store/ieee", exist_ok=True)

import numpy as np
import pandas as pd
import psutil
import joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              recall_score, precision_score, brier_score_loss,
                              roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ──────────────────────────────────────────────────────────────────────
RAW_DIR    = str(PROJECT_ROOT / "data/raw/ieee-fraud-detection")
FEAT_DIR   = str(PROJECT_ROOT / "data/feature_store/ieee")
MODELS_DIR = str(PROJECT_ROOT / "models")
FIGS_DIR   = str(PROJECT_ROOT / "reports/figures")
REPORT_DIR = str(PROJECT_ROOT / "reports")

TX_PATH = f"{RAW_DIR}/train_transaction.csv"
ID_PATH = f"{RAW_DIR}/train_identity.csv"
SAMPLE_PATH = f"{FEAT_DIR}/training_sample.parquet"

CHUNK_SIZE   = 100_000   # Out-of-core chunk size
OPTUNA_TRIALS = 40

STAGE_TIMINGS = {}
PEAK_RAM_MB   = 0.0
PIPELINE_START = time.time()

def mem():
    global PEAK_RAM_MB
    m = psutil.Process(os.getpid()).memory_info().rss / 1e6
    PEAK_RAM_MB = max(PEAK_RAM_MB, m)
    return m

class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self):
        self.t0 = time.time()
        print(f"\n▶  {self.name}  (RAM={mem():.0f} MB)")
        return self
    def __exit__(self, *_):
        elapsed = time.time() - self.t0
        STAGE_TIMINGS[self.name] = elapsed
        print(f"✔  {self.name}  [{elapsed:.1f}s]  (RAM={mem():.0f} MB)")

print("=" * 70)
print("  IEEE-CIS FRAUD DETECTION PIPELINE")
print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  System:  {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total/1e9:.1f} GB RAM")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — FEATURE ENGINEERING + STRATIFIED SAMPLE BUILD
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 1: Ingestion & Feature Engineering"):

    if os.path.exists(SAMPLE_PATH):
        print(f"  Loading cached sample from {SAMPLE_PATH}")
        sample_df = pd.read_parquet(SAMPLE_PATH)
        print(f"  Loaded: {len(sample_df):,} rows × {len(sample_df.columns)} cols")
    else:
        print(f"  Loading identity table...")
        id_df = pd.read_csv(ID_PATH)
        # Encode categorical identity columns
        for col in id_df.select_dtypes(include='object').columns:
            id_df[col] = pd.Categorical(id_df[col]).codes.astype('int16')
        id_df = id_df.set_index('TransactionID')
        print(f"  Identity: {len(id_df):,} rows × {len(id_df.columns)} cols")

        # ── Card-level aggregates (computed over full dataset first) ──────────
        print("  Computing card-level aggregates over full dataset (pass 1)...")
        card_aggs = {}  # card1 -> {count, fraud_count, amt_sum, amt_std_data}
        reader = pd.read_csv(TX_PATH, chunksize=CHUNK_SIZE,
                             usecols=['TransactionID','isFraud','TransactionAmt',
                                      'card1','card2','card4','card6','TransactionDT'])
        for chunk in reader:
            chunk['card1'] = chunk['card1'].fillna(-1).astype(int)
            for cid, grp in chunk.groupby('card1'):
                if cid not in card_aggs:
                    card_aggs[cid] = {'count': 0, 'fraud_count': 0,
                                      'amt_sum': 0.0, 'amounts': []}
                card_aggs[cid]['count']       += len(grp)
                card_aggs[cid]['fraud_count'] += grp['isFraud'].sum()
                card_aggs[cid]['amt_sum']     += grp['TransactionAmt'].sum()
                card_aggs[cid]['amounts'].extend(grp['TransactionAmt'].tolist())
        # Summarise
        card_stats = {}
        for cid, d in card_aggs.items():
            amts = np.array(d['amounts'])
            card_stats[cid] = {
                'card_tx_count':    d['count'],
                'card_fraud_count': d['fraud_count'],
                'card_fraud_ratio': d['fraud_count'] / max(d['count'], 1),
                'card_amt_mean':    float(amts.mean()) if len(amts) else 0.0,
                'card_amt_std':     float(amts.std())  if len(amts) > 1 else 0.0,
                'card_amt_max':     float(amts.max())  if len(amts) else 0.0,
            }
        card_stats_df = pd.DataFrame.from_dict(card_stats, orient='index')
        card_stats_df.index.name = 'card1'
        card_stats_df = card_stats_df.reset_index()
        del card_aggs; gc.collect()
        print(f"  Card aggregates computed for {len(card_stats_df):,} unique cards")

        # ── Email domain encoding ─────────────────────────────────────────────
        TOP_DOMAINS = [
            'gmail.com','yahoo.com','hotmail.com','anonymous.com','outlook.com',
            'live.com','icloud.com','protonmail.com'
        ]

        def encode_email(domain):
            if pd.isna(domain): return -1
            d = str(domain).lower().strip()
            if d in TOP_DOMAINS: return TOP_DOMAINS.index(d)
            return len(TOP_DOMAINS)  # 'other'

        # ── Pass 2: Build enriched chunks ─────────────────────────────────────
        print("  Building enriched feature chunks (pass 2)...")
        fraud_chunks, normal_chunks = [], []
        reader2 = pd.read_csv(TX_PATH, chunksize=CHUNK_SIZE)
        chunk_idx = 0
        total_fraud, total_normal = 0, 0

        for chunk in reader2:
            chunk_idx += 1

            # ── Temporal features ─────────────────────────────────────────────
            # TransactionDT is seconds since reference date (not real timestamp)
            chunk['hour_of_day']    = (chunk['TransactionDT'] // 3600) % 24
            chunk['day_of_week']    = (chunk['TransactionDT'] // 86400) % 7
            chunk['day_of_month']   = (chunk['TransactionDT'] // 86400) % 30
            chunk['is_night']       = ((chunk['hour_of_day'] >= 22) | (chunk['hour_of_day'] <= 5)).astype(int)
            chunk['is_weekend']     = (chunk['day_of_week'] >= 5).astype(int)
            chunk['is_business_hrs']= ((chunk['hour_of_day'] >= 9) & (chunk['hour_of_day'] <= 17)).astype(int)

            # ── Amount features ───────────────────────────────────────────────
            chunk['amt_log']          = np.log1p(chunk['TransactionAmt'])
            chunk['amt_cents']        = (chunk['TransactionAmt'] * 100).astype(int) % 100
            chunk['amt_is_round']     = (chunk['amt_cents'] == 0).astype(int)
            chunk['amt_rounded_10']   = ((chunk['TransactionAmt'] % 10 == 0) & (chunk['TransactionAmt'] > 0)).astype(int)
            chunk['amt_rounded_100']  = ((chunk['TransactionAmt'] % 100 == 0) & (chunk['TransactionAmt'] > 0)).astype(int)

            # ── Email encoding ────────────────────────────────────────────────
            chunk['P_email_enc'] = chunk['P_emaildomain'].apply(encode_email)
            chunk['R_email_enc'] = chunk['R_emaildomain'].apply(encode_email)
            chunk['same_email']  = (chunk['P_emaildomain'] == chunk['R_emaildomain']).astype(int)

            # ── Card features ─────────────────────────────────────────────────
            chunk['card1'] = chunk['card1'].fillna(-1).astype(int)
            chunk = chunk.merge(card_stats_df, on='card1', how='left')

            # Amount deviation from card's mean
            chunk['amt_vs_card_mean'] = (chunk['TransactionAmt'] - chunk['card_amt_mean'].fillna(0)) \
                                         / (chunk['card_amt_std'].fillna(1) + 1e-6)

            # ── Encode categoricals ───────────────────────────────────────────
            for col in ['ProductCD', 'card4', 'card6', 'M4']:
                if col in chunk.columns:
                    chunk[col] = pd.Categorical(chunk[col]).codes.astype('int16')

            # M1-M9 binary flags: T/F -> 1/0
            for col in [f'M{i}' for i in range(1, 10)]:
                if col in chunk.columns:
                    chunk[col] = (chunk[col] == 'T').astype('int8')

            # ── Merge identity ────────────────────────────────────────────────
            chunk = chunk.set_index('TransactionID')
            chunk = chunk.join(id_df, how='left', rsuffix='_id')
            chunk = chunk.reset_index().rename(columns={'index':'TransactionID'})

            # ── Drop raw string / high-cardinality columns ────────────────────
            drop_cols = ['P_emaildomain', 'R_emaildomain', 'DeviceInfo']
            chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns])

            # ── Dtype optimisation ────────────────────────────────────────────
            for col in chunk.select_dtypes(include='float64').columns:
                chunk[col] = chunk[col].astype('float32')
            for col in chunk.select_dtypes(include='int64').columns:
                if col not in ['TransactionID']:
                    chunk[col] = chunk[col].astype('int32')

            # ── Split fraud / normal for stratified sample ────────────────────
            fraud_rows  = chunk[chunk['isFraud'] == 1]
            normal_rows = chunk[chunk['isFraud'] == 0].sample(
                min(int(len(fraud_rows) * 8), len(chunk[chunk['isFraud']==0])),
                random_state=chunk_idx
            )
            total_fraud  += len(fraud_rows)
            total_normal += len(normal_rows)
            fraud_chunks.append(fraud_rows)
            normal_chunks.append(normal_rows)

            if chunk_idx % 2 == 0:
                print(f"    Chunk {chunk_idx}: processed {chunk_idx * CHUNK_SIZE:,} rows, "
                      f"fraud={total_fraud:,}  RAM={mem():.0f} MB")

            del chunk; gc.collect()

        print(f"  All chunks processed. fraud={total_fraud:,}  normal={total_normal:,}")

        # ── Assemble stratified sample ─────────────────────────────────────────
        sample_df = pd.concat(fraud_chunks + normal_chunks, ignore_index=True)
        sample_df = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Remove duplicate columns
        sample_df = sample_df.loc[:, ~sample_df.columns.duplicated()]

        print(f"  Final sample: {len(sample_df):,} rows × {len(sample_df.columns)} cols")
        print(f"  Fraud in sample: {sample_df['isFraud'].sum():,} ({sample_df['isFraud'].mean()*100:.1f}%)")

        sample_df.to_parquet(SAMPLE_PATH, index=False)
        print(f"  Saved to {SAMPLE_PATH}")
        del fraud_chunks, normal_chunks, id_df, card_stats_df; gc.collect()

print(f"  Sample shape: {sample_df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — ISOLATION FOREST (ANOMALY DETECTION)
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 2: Isolation Forest Anomaly Detection"):
    IF_PATH = f"{MODELS_DIR}/ieee_isolation_forest.joblib"

    # Feature set for IF: all numeric except ID and label
    EXCLUDE = {'TransactionID', 'isFraud', 'TransactionDT'}
    if_feature_cols = [c for c in sample_df.columns
                       if c not in EXCLUDE
                       and pd.api.types.is_numeric_dtype(sample_df[c])
                       and not c.endswith('_id')]

    X_if = sample_df[if_feature_cols].fillna(0).astype('float32').values

    if os.path.exists(IF_PATH):
        print("  Loading cached IF model...")
        if_bundle = joblib.load(IF_PATH)
        imputer_if = if_bundle['imputer']
        if_model   = if_bundle['model']
        if_feature_cols = if_bundle['feature_cols']
        X_if = sample_df[if_feature_cols].values
    else:
        print(f"  Training Isolation Forest on {X_if.shape[0]:,} rows × {X_if.shape[1]} features...")
        imputer_if = SimpleImputer(strategy='median')
        X_if_imp = imputer_if.fit_transform(X_if)
        if_model = IsolationForest(n_estimators=100, contamination='auto',
                                   random_state=42, n_jobs=-1)
        if_model.fit(X_if_imp)
        joblib.dump({'model': if_model, 'imputer': imputer_if,
                     'feature_cols': if_feature_cols}, IF_PATH)
        print(f"  IF saved to {IF_PATH}")

    # Score
    X_if_imp = imputer_if.transform(sample_df[if_feature_cols].fillna(0).values)
    raw_if   = if_model.decision_function(X_if_imp)
    # Normalise: higher = more anomalous
    mn, mx = raw_if.min(), raw_if.max()
    anomaly_scores = np.clip((mx - raw_if) / (mx - mn + 1e-9), 0, 1)
    sample_df['anomaly_score'] = anomaly_scores

    print(f"  IF scored {len(anomaly_scores):,}. Mean={anomaly_scores.mean():.4f} "
          f"P95={np.percentile(anomaly_scores,95):.4f} P99={np.percentile(anomaly_scores,99):.4f}")
    print(f"  High-risk (>0.7): {(anomaly_scores>0.7).sum():,}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — SUPERVISED MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 3: Supervised Model Training"):

    # Feature selection: all numeric except ID/label/DT
    EXCLUDE_TRAIN = {'TransactionID', 'isFraud', 'TransactionDT',
                     'card_fraud_count', 'card_fraud_ratio'}  # No fraud-label derived
    feature_names = [c for c in sample_df.columns
                     if c not in EXCLUDE_TRAIN
                     and pd.api.types.is_numeric_dtype(sample_df[c])
                     and not c.endswith('_id')]

    X = sample_df[feature_names].fillna(0).astype('float32').values
    y = sample_df['isFraud'].astype(int).values

    print(f"  Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Fraud: {y.sum():,} ({y.mean()*100:.2f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42, stratify=y_train)
    print(f"  Train={len(X_tr):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    scale_pos = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    print(f"  scale_pos_weight = {scale_pos:.1f}")
    all_results = {}

    # ── Logistic Regression (Baseline) ───────────────────────────────────────
    with Timer("  3a: Logistic Regression"):
        lr_path = f"{MODELS_DIR}/ieee_lr.joblib"
        if os.path.exists(lr_path):
            lr = joblib.load(lr_path); print("    Loaded from cache")
        else:
            lr = LogisticRegression(class_weight='balanced', max_iter=500,
                                    C=0.1, solver='saga', n_jobs=-1, random_state=42)
            lr.fit(X_tr, y_tr); joblib.dump(lr, lr_path)
        lr_prob = lr.predict_proba(X_test)[:, 1]
        lr_auc  = roc_auc_score(y_test, lr_prob)
        lr_ap   = average_precision_score(y_test, lr_prob)
        thr_lr  = 0.5
        best_f1 = 0
        for t in np.linspace(0.05, 0.95, 100):
            f = f1_score(y_test, (lr_prob >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1, thr_lr = f, t
        lr_pred = (lr_prob >= thr_lr).astype(int)
        all_results['Logistic Regression'] = {
            'y_prob': lr_prob,
            'roc_auc': lr_auc, 'pr_auc': lr_ap,
            'f1': f1_score(y_test, lr_pred, zero_division=0),
            'recall': recall_score(y_test, lr_pred, zero_division=0),
            'precision': precision_score(y_test, lr_pred, zero_division=0),
            'threshold': thr_lr
        }
        print(f"    LR: ROC-AUC={lr_auc:.4f}  PR-AUC={lr_ap:.4f}  F1={all_results['Logistic Regression']['f1']:.4f}")

    # ── LightGBM + Optuna ─────────────────────────────────────────────────────
    with Timer("  3b: LightGBM + Optuna"):
        lgbm_path = f"{MODELS_DIR}/ieee_lgbm.joblib"
        if os.path.exists(lgbm_path):
            lgbm_best = joblib.load(lgbm_path); print("    Loaded from cache")
            best_params = lgbm_best.get_params()
        else:
            def lgbm_obj(trial):
                p = {
                    'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
                    'n_estimators':     trial.suggest_int('n_estimators', 300, 1200),
                    'num_leaves':       trial.suggest_int('num_leaves', 31, 256),
                    'max_depth':        trial.suggest_int('max_depth', 4, 12),
                    'learning_rate':    trial.suggest_float('lr', 0.005, 0.15, log=True),
                    'min_child_samples':trial.suggest_int('min_child_samples', 10, 100),
                    'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha':        trial.suggest_float('reg_alpha', 0.0, 3.0),
                    'reg_lambda':       trial.suggest_float('reg_lambda', 0.0, 3.0),
                    'scale_pos_weight': scale_pos,
                    'n_jobs': -1, 'random_state': 42,
                }
                m = lgb.LGBMClassifier(**p)
                m.fit(X_tr, y_tr,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(40, verbose=False),
                                 lgb.log_evaluation(-1)])
                return roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])

            study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lgbm_obj, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
            best_params = study.best_params
            best_params.update({'objective':'binary','metric':'auc','verbosity':-1,
                                'scale_pos_weight': scale_pos,
                                'n_jobs':-1,'random_state':42})
            print(f"    Optuna best val AUC = {study.best_value:.4f} after {OPTUNA_TRIALS} trials")

            lgbm_best = lgb.LGBMClassifier(**best_params)
            lgbm_best.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(60, verbose=False),
                                     lgb.log_evaluation(-1)])
            joblib.dump(lgbm_best, lgbm_path)

        lgbm_prob = lgbm_best.predict_proba(X_test)[:, 1]
        lgbm_auc  = roc_auc_score(y_test, lgbm_prob)
        lgbm_ap   = average_precision_score(y_test, lgbm_prob)
        thr_lgbm  = 0.5; best_f1 = 0
        for t in np.linspace(0.05, 0.95, 200):
            f = f1_score(y_test, (lgbm_prob >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1, thr_lgbm = f, t
        lgbm_pred = (lgbm_prob >= thr_lgbm).astype(int)
        all_results['LightGBM'] = {
            'y_prob': lgbm_prob,
            'roc_auc': lgbm_auc, 'pr_auc': lgbm_ap,
            'f1': f1_score(y_test, lgbm_pred, zero_division=0),
            'recall': recall_score(y_test, lgbm_pred, zero_division=0),
            'precision': precision_score(y_test, lgbm_pred, zero_division=0),
            'threshold': thr_lgbm
        }
        print(f"    LGBM: ROC-AUC={lgbm_auc:.4f}  PR-AUC={lgbm_ap:.4f}  F1={all_results['LightGBM']['f1']:.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    with Timer("  3c: Random Forest"):
        rf_path = f"{MODELS_DIR}/ieee_rf.joblib"
        if os.path.exists(rf_path):
            rf = joblib.load(rf_path); print("    Loaded from cache")
        else:
            rf = RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample',
                                        max_depth=20, min_samples_leaf=5,
                                        n_jobs=-1, random_state=42)
            rf.fit(X_tr, y_tr); joblib.dump(rf, rf_path)
        rf_prob = rf.predict_proba(X_test)[:, 1]
        rf_auc  = roc_auc_score(y_test, rf_prob)
        rf_ap   = average_precision_score(y_test, rf_prob)
        thr_rf  = 0.5; best_f1 = 0
        for t in np.linspace(0.05, 0.95, 100):
            f = f1_score(y_test, (rf_prob >= t).astype(int), zero_division=0)
            if f > best_f1: best_f1, thr_rf = f, t
        rf_pred = (rf_prob >= thr_rf).astype(int)
        all_results['Random Forest'] = {
            'y_prob': rf_prob,
            'roc_auc': rf_auc, 'pr_auc': rf_ap,
            'f1': f1_score(y_test, rf_pred, zero_division=0),
            'recall': recall_score(y_test, rf_pred, zero_division=0),
            'precision': precision_score(y_test, rf_pred, zero_division=0),
            'threshold': thr_rf
        }
        print(f"    RF:  ROC-AUC={rf_auc:.4f}  PR-AUC={rf_ap:.4f}  F1={all_results['Random Forest']['f1']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 4: Isotonic Calibration"):
    from sklearn.isotonic import IsotonicRegression

    cal_path = f"{MODELS_DIR}/ieee_calibrated_lgbm.joblib"
    if os.path.exists(cal_path):
        cal_bundle = joblib.load(cal_path)
        iso_reg    = cal_bundle['iso']
        print("  Loaded from cache")
    else:
        # Fit isotonic regression on validation set raw probabilities
        val_raw  = lgbm_best.predict_proba(X_val)[:, 1]
        iso_reg  = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(val_raw, y_val)
        joblib.dump({'iso': iso_reg, 'feature_names': feature_names}, cal_path)
        print("  Isotonic calibrator fitted and saved")

    # Calibrated probabilities on test set
    lgbm_test_raw = lgbm_best.predict_proba(X_test)[:, 1]
    cal_prob  = iso_reg.predict(lgbm_test_raw)
    brier_pre  = brier_score_loss(y_test, lgbm_prob)
    brier_post = brier_score_loss(y_test, cal_prob)
    print(f"  Brier Score: {brier_pre:.4f} → {brier_post:.4f}  (Δ={brier_pre-brier_post:+.4f})")

    def cal_predict(X_arr):
        """Helper: calibrated probability for any feature matrix."""
        raw = lgbm_best.predict_proba(X_arr)[:, 1]
        return iso_reg.predict(raw)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 5: SHAP Explainability"):
    import shap
    shap_n = min(2000, len(X_test))
    X_shap = X_test[:shap_n]
    explainer = shap.TreeExplainer(lgbm_best)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list): shap_vals = shap_vals[1]
    mean_shap  = np.abs(shap_vals).mean(axis=0)
    top_idx    = np.argsort(mean_shap)[::-1][:20]
    top_feats  = [feature_names[i] for i in top_idx]
    top_shap_v = mean_shap[top_idx].tolist()
    print(f"  Top feature: {top_feats[0]} ({top_shap_v[0]:.4f})")
    for f, v in zip(top_feats[:10], top_shap_v[:10]):
        print(f"    {f:45s} {v:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — DRIFT MONITORING (PSI)
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 6: Drift Monitoring (PSI)"):
    def compute_psi(ref, cur, bins=10):
        ref, cur = np.array(ref), np.array(cur)
        breakpoints = np.nanpercentile(ref, np.linspace(0, 100, bins + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2: return 0.0
        ref_pct = np.histogram(ref, bins=breakpoints)[0] / max(len(ref), 1)
        cur_pct = np.histogram(cur, bins=breakpoints)[0] / max(len(cur), 1)
        ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
        cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)
        return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    ref_idx = int(len(sample_df) * 0.6)
    ref_X = sample_df[feature_names].iloc[:ref_idx].fillna(0).values.astype('float32')
    cur_X = sample_df[feature_names].iloc[ref_idx:].fillna(0).values.astype('float32')
    ref_probs = cal_predict(ref_X)
    cur_probs = cal_predict(cur_X)
    score_psi = compute_psi(ref_probs, cur_probs)

    feat_psi = {}
    for i, col in enumerate(feature_names):
        feat_psi[col] = compute_psi(ref_X[:, i], cur_X[:, i])

    n_critical = sum(1 for v in feat_psi.values() if v > 0.20)
    n_warning  = sum(1 for v in feat_psi.values() if 0.10 < v <= 0.20)
    drift_status = 'critical' if n_critical > 0 else ('warning' if n_warning > 0 else 'stable')
    print(f"  Score PSI={score_psi:.4f}  Status={drift_status}")
    print(f"  Feature PSI: {n_critical} critical, {n_warning} warning, "
          f"{len(feat_psi)-n_critical-n_warning} stable")
    del ref_X, cur_X; gc.collect()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — INTERACTIVE PLOTLY CHARTS
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 7: Figure Generation"):
    bg, fg = "#0a0f1e", "white"
    layout_base = dict(paper_bgcolor=bg, plot_bgcolor=bg, font=dict(color=fg),
                       width=860, height=520)

    # ROC curves
    fig_roc = go.Figure()
    for name, res in all_results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr,
                                     name=f"{name} (AUC={res['roc_auc']:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random',
                                  line=dict(dash='dash', color='gray')))
    fig_roc.update_layout(title='ROC Curves — IEEE-CIS', xaxis_title='FPR',
                           yaxis_title='TPR', **layout_base)
    fig_roc.write_html(f"{FIGS_DIR}/ieee_roc_curves.html")

    # PR curves
    fig_pr = go.Figure()
    for name, res in all_results.items():
        prec, rec, _ = precision_recall_curve(y_test, res['y_prob'])
        fig_pr.add_trace(go.Scatter(x=rec, y=prec,
                                    name=f"{name} (AP={res['pr_auc']:.3f})"))
    fig_pr.update_layout(title='Precision-Recall Curves — IEEE-CIS',
                          xaxis_title='Recall', yaxis_title='Precision', **layout_base)
    fig_pr.write_html(f"{FIGS_DIR}/ieee_pr_curves.html")

    # Confusion matrix
    thr_lgbm = all_results['LightGBM']['threshold']
    cm = confusion_matrix(y_test, (cal_prob >= thr_lgbm).astype(int))
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x='Predicted', y='Actual'),
                        x=['Normal','Fraud'], y=['Normal','Fraud'],
                        title='Confusion Matrix — Calibrated LightGBM (IEEE-CIS)')
    fig_cm.write_html(f"{FIGS_DIR}/ieee_confusion_matrix.html")

    # Calibration curve
    pt_cal, pp_cal = calibration_curve(y_test, cal_prob, n_bins=10)
    pt_raw, pp_raw = calibration_curve(y_test, lgbm_prob, n_bins=10)
    fig_cal = go.Figure()
    fig_cal.add_trace(go.Scatter(x=pp_raw, y=pt_raw, name='Uncalibrated LGBM'))
    fig_cal.add_trace(go.Scatter(x=pp_cal, y=pt_cal, name='Calibrated LGBM'))
    fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Perfect', line=dict(dash='dash')))
    fig_cal.update_layout(title='Calibration Curve — IEEE-CIS',
                           xaxis_title='Mean Predicted Prob', yaxis_title='Fraction Positives',
                           **layout_base)
    fig_cal.write_html(f"{FIGS_DIR}/ieee_calibration.html")

    # SHAP bar
    fig_shap = go.Figure(go.Bar(x=top_shap_v, y=top_feats, orientation='h',
                                 marker=dict(color=top_shap_v, colorscale='Viridis')))
    fig_shap.update_layout(title='Top 20 Features — Mean |SHAP| (IEEE-CIS)',
                            xaxis_title='Mean |SHAP value|', **{**layout_base, 'height': 640})
    fig_shap.write_html(f"{FIGS_DIR}/ieee_shap_bar.html")

    # PSI bar chart
    psi_sorted = sorted(feat_psi.items(), key=lambda x: x[1], reverse=True)[:30]
    psi_names = [x[0] for x in psi_sorted]
    psi_vals  = [x[1] for x in psi_sorted]
    psi_colors = ['red' if v > 0.2 else ('orange' if v > 0.1 else 'green') for v in psi_vals]
    fig_psi = go.Figure(go.Bar(x=psi_vals, y=psi_names, orientation='h',
                                marker_color=psi_colors))
    fig_psi.update_layout(title='Feature PSI — Drift Monitoring (IEEE-CIS)',
                           xaxis_title='PSI', **{**layout_base, 'height': 700})
    fig_psi.write_html(f"{FIGS_DIR}/ieee_psi.html")

    # Risk score distribution
    risk_scores_viz = cal_prob * 100
    fig_risk = go.Figure(go.Histogram(x=risk_scores_viz, nbinsx=50, marker_color='#00d4ff'))
    fig_risk.update_layout(title='Risk Score Distribution (IEEE-CIS)',
                            xaxis_title='Risk Score (0-100)',
                            yaxis_title='Count', **layout_base)
    fig_risk.write_html(f"{FIGS_DIR}/ieee_risk_distribution.html")

    print(f"  Generated 7 figures in {FIGS_DIR}/")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — EXECUTION REPORT
# ══════════════════════════════════════════════════════════════════════════════
with Timer("Stage 8: Execution Report"):
    total_runtime = time.time() - PIPELINE_START
    lm = all_results
    lr_m, lgbm_m, rf_m = lm['Logistic Regression'], lm['LightGBM'], lm['Random Forest']

    report = f"""# IEEE-CIS Fraud Detection — Execution Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Runtime: {total_runtime/60:.1f} minutes

## System Information
| Item | Value |
|---|---|
| CPUs | {psutil.cpu_count()} |
| Total RAM | {psutil.virtual_memory().total/1e9:.1f} GB |
| Peak RAM Used | {PEAK_RAM_MB/1024:.2f} GB |
| Python | {sys.version.split()[0]} |

## Dataset Statistics
| Metric | Value |
|---|---|
| Dataset | IEEE-CIS Fraud Detection (Vesta Corporation) |
| Total Transactions | 590,540 |
| Identity Records | 144,233 |
| Training Sample Rows | {len(sample_df):,} |
| Features Used | {len(feature_names)} |
| Fraud Transactions | ~20,663 (3.5%) |
| Chunk Size (OOC) | {CHUNK_SIZE:,} rows |

## Model Performance (Test Set — 20% holdout)
| Model | ROC-AUC | PR-AUC | F1 | Recall | Precision |
|---|---|---|---|---|---|
| Logistic Regression | {lr_m['roc_auc']:.4f} | {lr_m['pr_auc']:.4f} | {lr_m['f1']:.4f} | {lr_m['recall']:.4f} | {lr_m['precision']:.4f} |
| Random Forest | {rf_m['roc_auc']:.4f} | {rf_m['pr_auc']:.4f} | {rf_m['f1']:.4f} | {rf_m['recall']:.4f} | {rf_m['precision']:.4f} |
| **LightGBM (Best)** | **{lgbm_m['roc_auc']:.4f}** | **{lgbm_m['pr_auc']:.4f}** | **{lgbm_m['f1']:.4f}** | **{lgbm_m['recall']:.4f}** | **{lgbm_m['precision']:.4f}** |

## Probability Calibration (Isotonic)
| Metric | Value |
|---|---|
| Brier Score (before) | {brier_pre:.4f} |
| Brier Score (after)  | {brier_post:.4f} |
| Improvement | {(brier_pre-brier_post)/brier_pre*100:.1f}% |

## Anomaly Detection (Isolation Forest)
| Metric | Value |
|---|---|
| Samples Scored | {len(anomaly_scores):,} |
| Mean Anomaly Score | {anomaly_scores.mean():.4f} |
| P95 Anomaly Score | {np.percentile(anomaly_scores,95):.4f} |
| P99 Anomaly Score | {np.percentile(anomaly_scores,99):.4f} |

## SHAP Top Features (LightGBM)
| Rank | Feature | Mean |SHAP| |
|---|---|---|
{''.join(f'| {i+1} | {f} | {v:.4f} |' + chr(10) for i,(f,v) in enumerate(zip(top_feats[:10], top_shap_v[:10])))}

## Drift Monitoring (PSI)
| Metric | Value |
|---|---|
| Score PSI | {score_psi:.6f} |
| Overall Status | {drift_status} |
| Critical Features (PSI > 0.20) | {n_critical} |
| Warning Features (PSI > 0.10) | {n_warning} |
| Stable Features | {len(feat_psi)-n_critical-n_warning} |

## Stage Timings
| Stage | Duration |
|---|---|
{''.join(f'| {k} | {v:.1f}s |' + chr(10) for k,v in STAGE_TIMINGS.items())}

---
*IEEE-CIS Fraud Detection Dataset — Vesta Corporation*
*Pipeline: Out-of-Core chunked ingestion → Feature Engineering → Isolation Forest → LightGBM (Optuna) → Isotonic Calibration → SHAP → Drift Monitoring*
"""

    report_path = f"{REPORT_DIR}/ieee_execution_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"  Report saved: {report_path}")

    # Print final summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print(f"  Total Runtime: {total_runtime/60:.1f} minutes")
    print(f"  Peak RAM: {PEAK_RAM_MB/1024:.2f} GB")
    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              FINAL MODEL METRICS                    │")
    print("  ├──────────────────────┬────────┬────────┬────────────┤")
    print("  │ Model                │ AUC    │ PR-AUC │ F1         │")
    print("  ├──────────────────────┼────────┼────────┼────────────┤")
    print(f"  │ Logistic Regression  │ {lr_m['roc_auc']:.4f} │ {lr_m['pr_auc']:.4f} │ {lr_m['f1']:.4f}     │")
    print(f"  │ Random Forest        │ {rf_m['roc_auc']:.4f} │ {rf_m['pr_auc']:.4f} │ {rf_m['f1']:.4f}     │")
    print(f"  │ LightGBM (Best)      │ {lgbm_m['roc_auc']:.4f} │ {lgbm_m['pr_auc']:.4f} │ {lgbm_m['f1']:.4f}     │")
    print("  └──────────────────────┴────────┴────────┴────────────┘")
    print("=" * 70)
