import argparse, os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, brier_score_loss

def parse_scale(s):
    if not isinstance(s, str): return {}
    pairs = re.split(r';\s*', s.strip())
    mapping = {}
    for p in pairs:
        m = re.match(r'\s*([-\w]+)\s*=\s*(.+)', p)
        if m:
            k, v = m.group(1), m.group(2)
            try: k = int(k)
            except: pass
            mapping[k] = v
    return mapping

def apply_codebook(df, codebook_path):
    if not codebook_path or not os.path.exists(codebook_path): return df
    cb = pd.read_csv(codebook_path, sep=';')
    cb.columns = [c.strip().lower() for c in cb.columns]
    colmap = {c.lower(): c for c in df.columns}
    for _, row in cb.iterrows():
        col = str(row.get('variable name','')).strip().lower()
        scale = parse_scale(row.get('variable scale',''))
        if col and scale and col in colmap:
            orig = colmap[col]
            try:
                mapped = df[orig].map(scale)
                if mapped.notna().sum() > 0:
                    df[orig] = df[orig].where(mapped.isna(), mapped)
            except Exception:
                pass
    return df

def derive_labels(df):
    df = df.copy()
    # risk labels (proxies, not diagnoses)
    if 'cesd' in df.columns:
        df['y_dep'] = (pd.to_numeric(df['cesd'], errors='coerce') >= 16).astype(int)
    if 'stai_t' in df.columns:
        df['y_anx'] = (pd.to_numeric(df['stai_t'], errors='coerce') >= 45).astype(int)
    if 'mbi_ex' in df.columns:
        q75 = pd.to_numeric(df['mbi_ex'], errors='coerce').quantile(0.75)
        df['y_burn'] = (pd.to_numeric(df['mbi_ex'], errors='coerce') >= q75).astype(int)
    # language coarse group
    if 'glang' in df.columns:
        df['glang_group'] = df['glang'].astype(str).str.lower().apply(
            lambda x: 'German' if 'germ' in x else 'Non-German'
        )
    # drop obvious IDs if present
    for idc in ['id','Id','participant_id']:
        if idc in df.columns: df = df.drop(columns=[idc])
    return df

def build_model(X_train, y_train, num_cols, cat_cols, sample_weight=None):
    pre = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])
    clf = Pipeline([('pre', pre), ('lr', LogisticRegression(max_iter=300))])
    if sample_weight is not None:
        clf.fit(X_train, y_train, lr__sample_weight=sample_weight)
    else:
        clf.fit(X_train, y_train)
    return clf, pre

def metrics(y_true, y_pred, y_prob):
    out = {
        'AUC': roc_auc_score(y_true, y_prob),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall/TPR': recall_score(y_true, y_pred, zero_division=0),
        'Brier': brier_score_loss(y_true, y_prob)
    }
    return out

def group_report(df_eval, group_col):
    y_true = df_eval['y_true']; y_pred = df_eval['y_pred']; y_prob = df_eval['y_prob']
    base_pos = y_pred.mean()
    base_tpr = ((y_pred==1) & (y_true==1)).sum() / max(1,(y_true==1).sum())
    base_brier = brier_score_loss(y_true, y_prob)
    rows=[]
    for g, sub in df_eval.groupby(group_col):
        gt, gp, gb = sub['y_true'], sub['y_pred'], sub['y_prob']
        pos = gp.mean()
        tpr = ((gp==1) & (gt==1)).sum() / max(1,(gt==1).sum())
        brier = brier_score_loss(gt, gb)
        rows.append({
            group_col: g,
            'n': len(sub),
            'FlagRate': pos,
            'SPD': pos - base_pos,
            'TPR': tpr,
            'EOD': tpr - base_tpr,
            'Brier': brier,
            'Calib_Gap': brier - base_brier
        })
    return pd.DataFrame(rows).sort_values('SPD')

def plot_counts(df, col, outpng):
    vc = df[col].value_counts(dropna=False).sort_index()
    plt.figure()
    vc.plot(kind='bar')
    plt.title(f'{col} distribution')
    plt.xlabel(col); plt.ylabel('Count')
    plt.tight_layout(); plt.savefig(outpng); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='path to medteach.csv')
    ap.add_argument('--codebook', default='', help='optional codebook.csv')
    ap.add_argument('--target', default='y_dep', choices=['y_dep','y_anx','y_burn'])
    args = ap.parse_args()

    os.makedirs('outputs', exist_ok=True)
    os.makedirs('charts', exist_ok=True)

    df = pd.read_csv(args.data)
    df = apply_codebook(df, args.codebook)
    df = derive_labels(df)

    if args.target not in df.columns:
        raise SystemExit(f"Target {args.target} not found after label derivation.")

    y = df[args.target]
    X = df.drop(columns=[c for c in ['y_dep','y_anx','y_burn'] if c in df.columns])

    # Separate numeric / categorical
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.nunique()==2 else None
    )

    # Baseline
    clf, pre = build_model(X_train, y_train, num_cols, cat_cols)
    proba = clf.predict_proba(X_test)[:,1]
    pred = (proba >= 0.5).astype(int)
    base = metrics(y_test, pred, proba)
    pd.Series(base).to_csv('outputs/metrics_base.csv')
    print('Base metrics:', base)

    eval_df = X_test.copy()
    eval_df['y_true']=y_test; eval_df['y_pred']=pred; eval_df['y_prob']=proba

    groups = [c for c in ['sex','glang','glang_group','year'] if c in eval_df.columns]
    with pd.ExcelWriter('outputs/group_metrics_base.xlsx') as xw:
        for g in groups:
            rep = group_report(eval_df, g)
            rep.to_excel(xw, sheet_name=g, index=False)
            # counts chart
            plot_counts(X, g, f'charts/{g}_counts.png')

    # Simple mitigation: reweight by minority in 'sex' if present
    sample_weight = None
    if 'sex' in X_train.columns:
        counts = X_train['sex'].value_counts()
        inv = 1.0 / counts
        w = X_train['sex'].map(inv)
        sample_weight = (w / w.mean()).values

    clf2, _ = build_model(X_train, y_train, num_cols, cat_cols, sample_weight=sample_weight)
    proba2 = clf2.predict_proba(X_test)[:,1]
    pred2 = (proba2 >= 0.5).astype(int)
    rw = metrics(y_test, pred2, proba2)
    pd.Series(rw).to_csv('outputs/metrics_reweighted.csv')
    print('Reweighted metrics:', rw)

    eval_df2 = X_test.copy()
    eval_df2['y_true']=y_test; eval_df2['y_pred']=pred2; eval_df2['y_prob']=proba2
    with pd.ExcelWriter('outputs/group_metrics_reweighted.xlsx') as xw:
        for g in groups:
            rep = group_report(eval_df2, g)
            rep.to_excel(xw, sheet_name=g, index=False)

    print("Saved outputs/: metrics csv + Excel. Saved charts/ for counts.")

if __name__ == "__main__":
    main()