import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

df = pd.read_csv(BASE + 'master_features_clean.csv',
                 encoding='utf-8-sig', low_memory=False)
l1 = df[df['dataset'] == 'LUNG1'].dropna(
    subset=['survival_time_days', 'event']).copy()
l1['event'] = l1['event'].astype(int)
luad = df[df['dataset'] == 'LUAD-CT-Survival'].copy()
rad_cols = [c for c in df.columns if c.startswith('original_')]
emb_cols = [c for c in df.columns if c.startswith('emb_')]
y_te = (luad['survival_label'] == 'Short').astype(int).values

def pca_ft(tr, te, n):
    ix, sx = SimpleImputer(strategy='median'), StandardScaler()
    z_tr = sx.fit_transform(ix.fit_transform(tr))
    z_te = sx.transform(ix.transform(te))
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(z_tr), p.transform(z_te)

Xr_tr, Xr_te = pca_ft(l1[rad_cols].values, luad[rad_cols].values, 30)
Xe_tr, Xe_te = pca_ft(l1[emb_cols].values, luad[emb_cols].values, 50)
Xj_tr = np.hstack([Xr_tr, Xe_tr])
Xj_te = np.hstack([Xr_te, Xe_te])
y_tr = l1[['survival_time_days','event']].reset_index(drop=True)

def fit(X):
    d = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    d['T'] = y_tr['survival_time_days'].values
    d['E'] = y_tr['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d, duration_col='T', event_col='E')
    return cph

def boot_auc(h, y, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    vals = []
    n = len(h)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        vals.append(roc_auc_score(y[idx], h[idx]))
    return np.mean(vals), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

print('===== LUAD Long/Short 外验：AUC bootstrap 95% CI (n_boot=2000) =====')
for name, X_tr, X_te in [('影像组学', Xr_tr, Xr_te),
                          ('Deep',     Xe_tr, Xe_te),
                          ('联合',     Xj_tr, Xj_te)]:
    cph = fit(X_tr)
    h = cph.predict_partial_hazard(
        pd.DataFrame(X_te, columns=[f'f{i}' for i in range(X_te.shape[1])])
    ).values
    auc_raw = roc_auc_score(y_te, h)
    auc_flip = roc_auc_score(y_te, -h)
    m, lo, hi = boot_auc(h, y_te)
    print(f'{name:<10} AUC={auc_raw:.3f}  '
          f'95%CI=[{lo:.3f},{hi:.3f}]  '
          f'翻转后AUC={auc_flip:.3f}')

print('\n判读：')
print('  CI 下界 > 0.5  -> 显著优于随机')
print('  CI 跨 0.5       -> n=40 太小，不显著')
print('  翻转后AUC      -> 若显著>原始，说明方向反了')
print('完成')
