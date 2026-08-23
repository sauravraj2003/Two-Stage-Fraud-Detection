"""
init_ieee_hitl_db.py
====================
Initializes the HITL SQLite database with IEEE-CIS fraud data.
Stratified sample: all fraud rows + sample of normal rows.
Computes: anomaly_score (IF) → lgbm_prob → calibrated → risk fusion.
"""
import os, sys, sqlite3
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RAW_SAMPLE = PROJECT_ROOT / "data/feature_store/ieee/training_sample.parquet"
DB_PATH    = PROJECT_ROOT / "data/hitl/ieee_investigation_queue.db"
MODELS_DIR = PROJECT_ROOT / "models"

os.makedirs(DB_PATH.parent, exist_ok=True)

def main():
    print("Loading models...")
    lgbm_model  = joblib.load(MODELS_DIR / "ieee_lgbm.joblib")
    if_bundle   = joblib.load(MODELS_DIR / "ieee_isolation_forest.joblib")
    cal_bundle  = joblib.load(MODELS_DIR / "ieee_calibrated_lgbm.joblib")

    if_model    = if_bundle['model']
    if_imputer  = if_bundle['imputer']
    if_feat_cols= if_bundle['feature_cols']
    iso_reg     = cal_bundle['iso']
    feature_names = cal_bundle['feature_names']  # 448 LGBM features

    print("Loading training sample...")
    df = pd.read_parquet(RAW_SAMPLE)

    # ── Stratified sample ─────────────────────────────────────────────────────
    fraud_df  = df[df['isFraud'] == 1]
    normal_df = df[df['isFraud'] == 0].sample(min(2000, len(df[df['isFraud']==0])), random_state=42)
    sample    = pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Stratified sample: {len(fraud_df):,} fraud + {len(normal_df):,} normal = {len(sample):,} total")

    # ── Stage 1: Isolation Forest anomaly score ───────────────────────────────
    print("Computing Isolation Forest scores...")
    X_if     = sample[if_feat_cols].fillna(0).values.astype('float32')
    X_if_imp = if_imputer.transform(X_if)
    raw_if   = if_model.decision_function(X_if_imp)
    mn, mx   = raw_if.min(), raw_if.max()
    if_scores = np.clip((mx - raw_if) / (mx - mn + 1e-9), 0, 1)
    sample['anomaly_score'] = if_scores

    # ── Stage 2: LightGBM probabilities ──────────────────────────────────────
    print(f"Computing LightGBM probabilities ({len(feature_names)} features)...")
    X_lgbm     = sample[feature_names].fillna(0).values.astype('float32')
    lgbm_probs = lgbm_model.predict_proba(X_lgbm)[:, 1]

    # ── Stage 3: Isotonic calibration ────────────────────────────────────────
    cal_probs = iso_reg.predict(lgbm_probs)

    # ── Stage 4: Business rules ───────────────────────────────────────────────
    rule_scores = np.zeros(len(sample))
    rule_scores += np.where(sample['TransactionAmt'] > 500, 0.30, 0.0)
    rule_scores += np.where(sample['amt_is_round']   == 1,  0.15, 0.0)
    rule_scores += np.where(sample['is_night']       == 1,  0.10, 0.0)
    rule_scores = np.clip(rule_scores, 0, 1)

    # ── Stage 5: Risk fusion → 0-100 score ───────────────────────────────────
    risk_scores = np.clip((0.70 * cal_probs + 0.20 * if_scores + 0.10 * rule_scores) * 100, 0, 100)

    # ── Risk categories ───────────────────────────────────────────────────────
    def categorise(s):
        if s >= 75: return 'Critical'
        if s >= 50: return 'High'
        if s >= 25: return 'Medium'
        return 'Low'
    risk_cats = [categorise(s) for s in risk_scores]

    sample['lgbm_prob']     = lgbm_probs
    sample['cal_prob']      = cal_probs
    sample['if_score']      = if_scores
    sample['rule_score']    = rule_scores
    sample['risk_score']    = risk_scores
    sample['risk_category'] = risk_cats
    sample['status']        = 'Pending'

    # ── Write to SQLite ───────────────────────────────────────────────────────
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    print("Saving to SQLite...")
    sample.to_sql('transactions', conn, if_exists='replace', index=False)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS investigator_feedback (
            transaction_id  TEXT PRIMARY KEY,
            decision        TEXT,
            notes           TEXT,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

    print(f"\n✅  DB initialised: {DB_PATH}")
    print(f"   Total records: {len(sample):,}")
    print("   Risk category breakdown:")
    print(pd.Series(risk_cats).value_counts().to_string())
    print(f"\n   Fraud: {sample['isFraud'].sum():,}  |  Normal: {(sample['isFraud']==0).sum():,}")
    print(f"   Avg cal_prob (fraud):  {cal_probs[sample['isFraud'].values==1].mean():.4f}")
    print(f"   Avg cal_prob (normal): {cal_probs[sample['isFraud'].values==0].mean():.4f}")

if __name__ == '__main__':
    main()
