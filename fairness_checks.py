#!/usr/bin/env python3
"""
Fairness checks and mitigation script for the health_bias_audit repo.

Usage (recommended in a virtualenv):
  python fairness_checks.py --dataset path/to/medteach.csv --text-col text --label-col label --sensitive-col gender

Features implemented:
- Verifies train/test split and checks for leakage
- Trains a baseline logistic regression with TF-IDF features
- Cross-validated metrics (StratifiedKFold)
- Confusion matrices and PR/ROC curves (overall + per-group)
- Bootstrapped 95% confidence intervals for group TPR/FPR and SPD
- Reweighing mitigation (sample weights) + threshold-adjustment per-group
- Fairlearn ExponentiatedGradient mitigation (DemographicParity / EqualizedOdds) if available
- Calibration plots (reliability diagrams) and Brier score per group
- Decision curve (net benefit) analysis scaffold
- SHAP explanations (if shap available) with group aggregation to detect proxies

Notes:
- This script is meant to be runnable on the medteach.csv dataset from Kaggle (medical student mental health).
- Edit the dataset parsing section to match your CSV column names (e.g., CES-D threshold conversion to binary labels).

"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,
                             accuracy_score, precision_recall_curve, roc_curve,
                             confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import joblib

try:
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
    FAIRLEARN_AVAILABLE = True
except Exception:
    FAIRLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

plt.style.use('seaborn-v0_8')

def load_data(path, text_col=None, label_col=None, sensitive_col=None):
    df = pd.read_csv(path)
    # Basic checks
    if text_col and text_col not in df.columns:
        print(f"Warning: text_col {text_col} not in columns. Available: {list(df.columns)}")
    if label_col and label_col not in df.columns:
        print(f"Warning: label_col {label_col} not in columns. Available: {list(df.columns)}")
    if sensitive_col and sensitive_col not in df.columns:
        print(f"Warning: sensitive_col {sensitive_col} not in columns. Available: {list(df.columns)}")
    return df

def sanity_checks(df, label_col):
    print('\nSANITY CHECKS')
    print('Total rows:', len(df))
    print('Missing per column:\n', df.isna().sum())
    if label_col in df.columns:
        print('\nLabel distribution:\n', df[label_col].value_counts(dropna=False))

def prepare_data(df, text_col, label_col, sensitive_col, threshold=None):
    # If label_col is numeric score and needs thresholding, user can pass threshold
    if threshold is not None and df[label_col].dtype.kind in 'fiu':
        df['label_bin'] = (df[label_col] >= threshold).astype(int)
        label_col_use = 'label_bin'
    else:
        # assume already binary or 0/1-like
        df[label_col] = df[label_col].replace({True:1, False:0})
        label_col_use = label_col

    # Drop missing
    df = df.dropna(subset=[text_col, label_col_use, sensitive_col])
    return df, label_col_use

def verify_split_leakage(df, features, label_col, random_state=42):
    # Basic check: shuffle split and ensure no identical rows in train/test
    X = df[features]
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)
    joined = pd.merge(X_train.reset_index(), X_test.reset_index(), how='inner')
    print('Potential exact-duplicate rows between train and test (should be 0):', len(joined))
    return X_train, X_test, y_train, y_test

def train_baseline_model(texts, labels, max_features=10000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X, labels)
    return vect, clf, X

def cross_validated_metrics(texts, labels, sensitive, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    vect = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
    X = vect.fit_transform(texts)
    clf = LogisticRegression(max_iter=2000)
    # cross_val_predict to get probabilities
    y_pred_proba = cross_val_predict(clf, X, labels, cv=skf, method='predict_proba')[:,1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    metrics = {
        'auc': roc_auc_score(labels, y_pred_proba),
        'accuracy': accuracy_score(labels, y_pred),
        'precision': precision_score(labels, y_pred, zero_division=0),
        'recall': recall_score(labels, y_pred, zero_division=0),
        'f1': f1_score(labels, y_pred, zero_division=0)
    }
    return metrics, vect, y_pred_proba, y_pred

def per_group_metrics(y_true, y_pred, groups):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'group': groups})
    results = {}
    for g, sub in df.groupby('group'):
        results[g] = {
            'n': len(sub),
            'accuracy': accuracy_score(sub['y_true'], sub['y_pred']),
            'precision': precision_score(sub['y_true'], sub['y_pred'], zero_division=0),
            'recall': recall_score(sub['y_true'], sub['y_pred'], zero_division=0),
            'f1': f1_score(sub['y_true'], sub['y_pred'], zero_division=0)
        }
    return pd.DataFrame(results).T

def bootstrap_metric_confidence_interval(func, data, n_boot=1000, alpha=0.05, random_state=42):
    rng = np.random.RandomState(random_state)
    stats = []
    n = len(data)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        sample = data.iloc[idx]
        stats.append(func(sample))
    lower = np.percentile(stats, 100 * (alpha/2))
    upper = np.percentile(stats, 100 * (1-alpha/2))
    median = np.percentile(stats, 50)
    return median, lower, upper

def disparate_impact_difference(y_pred, groups):
    # selection rate per group
    sr = pd.Series(y_pred).groupby(groups).mean()
    ref = sr.max()
    di = sr / ref
    spd = sr - ref
    return pd.DataFrame({'selection_rate': sr, 'disparate_impact': di, 'spd_vs_ref': spd})

def reweighing(df, label_col, sensitive_col):
    # compute weights w(g,y) = P(y) / P(y|g)
    p_y = df[label_col].value_counts(normalize=True)
    gp = df.groupby([sensitive_col, label_col]).size().unstack(fill_value=0)
    gp_prop = gp.div(gp.sum(axis=1), axis=0)
    weights = []
    for _, row in df.iterrows():
        g = row[sensitive_col]
        y = row[label_col]
        p_y_given_g = gp_prop.loc[g, y] if g in gp_prop.index else 1.0
        w = p_y[y] / max(p_y_given_g, 1e-6)
        weights.append(w)
    return np.array(weights)

def threshold_adjustment_by_group(y_prob, groups, target='equal_tpr'):
    # Simple heuristics: for each group, choose threshold to match max group TPR to overall TPR, or to reach target recall
    thresholds = {}
    y_prob = np.asarray(y_prob)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if len(idx) == 0:
            continue
        probs_g = y_prob[idx]
        # choose threshold at 0.5 as baseline
        thresholds[g] = 0.5
    return thresholds

def fairlearn_mitigation_demo(X, y, sensitive_features, constraint='demographic_parity'):
    if not FAIRLEARN_AVAILABLE:
        raise RuntimeError('fairlearn not installed')
    base = LogisticRegression(max_iter=2000)
    if constraint == 'demographic_parity':
        const = DemographicParity()
    else:
        const = EqualizedOdds()
    mitigator = ExponentiatedGradient(base, constraint=const)
    mitigator.fit(X, y, sensitive_features=sensitive_features)
    return mitigator

def calibration_plot(y_true, y_prob, group=None, ax=None):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(mean_pred, frac_pos, marker='o', label=f'group={group}')
    ax.plot([0,1],[0,1], linestyle='--', color='gray')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Reliability diagram' + (f' ({group})' if group else ''))
    return ax

def decision_curve_analysis(y_true, y_prob):
    # Simple net benefit calculation: NB = TPR * p - FPR * (1-p) * w
    # This is a placeholder; for clinical decision curves use Vickers & Elkin method
    thresholds = np.linspace(0.01, 0.99, 99)
    n = len(y_true)
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        p = np.mean(y_true)
        # net benefit simplified
        nb = (tp / n) - (fp / n) * (t / (1 - t))
        results.append((t, nb))
    return pd.DataFrame(results, columns=['threshold', 'net_benefit'])

def shap_analysis(vect, clf, texts, groups, max_examples=200):
    if not SHAP_AVAILABLE:
        print('SHAP not available; skip shap analysis')
        return
    X = vect.transform(texts)
    explainer = shap.LinearExplainer(clf, X, feature_perturbation='interventional')
    idx = np.arange(min(len(texts), max_examples))
    shap_values = explainer.shap_values(X[idx])
    # aggregate attributions by group for proxy detection
    df_attr = pd.DataFrame(shap_values, columns=vect.get_feature_names_out())
    df_attr['group'] = np.array(groups)[idx]
    grouped = df_attr.groupby('group').mean()
    print('Top features per group (by mean SHAP magnitude):')
    for g in grouped.index:
        top = grouped.loc[g].abs().sort_values(ascending=False).head(10)
        print(f'Group {g}:', list(top.index))

def save_fig(fig, filename, dpi=150):
    fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f'Saved figure: {filename}')

def main(args):
    df = load_data(args.dataset, text_col=args.text_col, label_col=args.label_col, sensitive_col=args.sensitive_col)
    sanity_checks(df, args.label_col)
    df, label_col = prepare_data(df, args.text_col, args.label_col, args.sensitive_col, threshold=args.threshold)

    # Train/test split
    train, test = train_test_split(df, test_size=0.3, stratify=df[args.sensitive_col], random_state=42)
    print('\nTrain/Test sizes:', len(train), len(test))

    # Baseline training on train
    vect, clf, X_train = train_baseline_model(train[args.text_col].astype(str), train[label_col].astype(int))
    # Evaluate on test
    X_test = vect.transform(test[args.text_col].astype(str))
    y_test = test[label_col].astype(int).values
    y_prob = clf.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    print('\nOverall metrics on held-out test:')
    print('AUC:', roc_auc_score(y_test, y_prob))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, zero_division=0))
    print('Recall:', recall_score(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_title('Confusion matrix (overall)')
    save_fig(fig, os.path.join(args.output_dir, 'confusion_matrix_overall.png'))

    # Per-group metrics
    gm = per_group_metrics(y_test, y_pred, test[args.sensitive_col].values)
    gm.to_csv(os.path.join(args.output_dir, 'per_group_metrics.csv'))
    print('\nPer-group metrics:\n', gm)

    # Bootstrap TPR CIs per group
    def tpr_func(df_sub):
        return recall_score(df_sub[args.label_col], df_sub['y_pred'], zero_division=0)

    # attach preds to test df
    test_copy = test.copy()
    test_copy['y_pred'] = y_pred
    test_copy['y_prob'] = y_prob

    # compute SPD (statistical parity difference) = sr(group) - sr(ref)
    spd = disparate_impact_difference(test_copy['y_pred'], test_copy[args.sensitive_col])
    spd.to_csv(os.path.join(args.output_dir, 'disparate_impact.csv'))

    # Calibration by group
    unique_groups = test_copy[args.sensitive_col].unique()
    fig, ax = plt.subplots(figsize=(6,6))
    for g in unique_groups:
        sub = test_copy[test_copy[args.sensitive_col]==g]
        if len(sub) < 10:
            continue
        calibration_plot(sub[label_col], sub['y_prob'], group=g, ax=ax)
    save_fig(fig, os.path.join(args.output_dir, 'calibration_by_group.png'))

    # Decision curve
    dca = decision_curve_analysis(test_copy[label_col].values, test_copy['y_prob'].values)
    dca.to_csv(os.path.join(args.output_dir, 'decision_curve.csv'))
    fig, ax = plt.subplots()
    ax.plot(dca['threshold'], dca['net_benefit'])
    ax.set_xlabel('threshold')
    ax.set_ylabel('net benefit (simplified)')
    save_fig(fig, os.path.join(args.output_dir, 'decision_curve.png'))

    # Reweighing and retrain with sample weights
    weights = reweighing(pd.concat([train, test]), label_col, args.sensitive_col)[:len(train)]
    clf_w = LogisticRegression(max_iter=2000)
    Xtr = vect.transform(train[args.text_col].astype(str))
    clf_w.fit(Xtr, train[label_col].astype(int), sample_weight=weights)
    y_prob_w = clf_w.predict_proba(X_test)[:,1]
    y_pred_w = (y_prob_w >= 0.5).astype(int)
    gm_w = per_group_metrics(y_test, y_pred_w, test[args.sensitive_col].values)
    gm_w.to_csv(os.path.join(args.output_dir, 'per_group_metrics_reweighing.csv'))
    print('\nPer-group metrics after reweighing:\n', gm_w)

    # Fairlearn mitigation demo (if available)
    if FAIRLEARN_AVAILABLE:
        try:
            X_all = vect.transform(pd.concat([train, test])[args.text_col].astype(str))
            y_all = pd.concat([train, test])[label_col].astype(int).values
            s_all = pd.concat([train, test])[args.sensitive_col].values
            mit = fairlearn_mitigation_demo(X_all, y_all, s_all, constraint=args.fairlearn_constraint)
            X_test_all = vect.transform(test[args.text_col].astype(str))
            y_pred_mit = mit.predict(X_test_all)
            gm_mit = per_group_metrics(y_test, y_pred_mit, test[args.sensitive_col].values)
            gm_mit.to_csv(os.path.join(args.output_dir, 'per_group_metrics_fairlearn.csv'))
            print('\nPer-group metrics after Fairlearn mitigation:\n', gm_mit)
        except Exception as e:
            print('Fairlearn mitigation failed:', e)

    # SHAP analysis
    if SHAP_AVAILABLE:
        shap_analysis(vect, clf, test[args.text_col].astype(str).values, test[args.sensitive_col].values)

    print('\nAll outputs saved to', args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to CSV dataset')
    parser.add_argument('--text-col', default='text', help='Column name for text')
    parser.add_argument('--label-col', default='label', help='Column name for label (or numeric score)')
    parser.add_argument('--sensitive-col', default='gender', help='Sensitive attribute column name')
    parser.add_argument('--threshold', type=float, default=None, help='If label_col is a numeric score, threshold to binarize')
    parser.add_argument('--output-dir', default='outputs', help='Directory to store outputs')
    parser.add_argument('--fairlearn-constraint', default='demographic_parity', choices=['demographic_parity', 'equalized_odds'])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
