# decay_decomp.py — 跨队列性能衰减分解
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

def load(feat_src, rad_pref, emb_pref):
    if feat_src == 'combat':
        d = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
        c = pd.read_csv(BASE + 'master_features_clean.csv',
                        encoding='utf-8-sig', low_memory=False)
        need = ['dataset','patient_id','age','gender','stage_clean','histology_clean']
        d = d.merge(c[need].drop_duplicates(['dataset','patient_id']),
                    on=['dataset','patient_id'], how='left')
    else:
        d = pd.read_csv(BASE + 'master_features_clean.csv',
                        encoding='utf-8-sig', low_memory=False)
    s = d[d['dataset'].isin(['LUNG1','Radiogenomics'])].dropna(
        subset=['survival_time_days','event']).copy()
    s['event'] = s['event'].astype(int)
    rad = [c for c in d.columns if c.startswith(rad_pref)]
    emb = [c for c in d.columns if c.startswith(emb_pref)]
    return s, rad, emb

def prep_clin(d):
    out = pd.DataFrame(index=d.index)
    out['age'] = d['age'].values
    out['gender_male'] = (d['gender']=='male').astype(int).values
    for s in ['I','II','III']:
        out['stage_'+s] = (d['stage_clean']==s).astype(int).values
    for h in ['adeno','squamous','large_cell','nos']:
        out['hist_'+h] = (d['histology_clean']==h).astype(int).values
    return out

def pca_ft(tr, te, n):
    ix, sx = SimpleImputer(strategy='median'), StandardScaler()
    z_tr = sx.fit_transform(ix.fit_transform(tr))
    z_te = sx.transform(ix.transform(te))
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(z_tr), p.transform(z_te)

def build(s, rad, emb):
    train = s[s['dataset']=='LUNG1'].copy()
    test  = s[s['dataset']=='Radiogenomics'].copy()
    imp, sc = SimpleImputer(strategy='median'), StandardScaler()
    Xc_tr = sc.fit_transform(imp.fit_transform(prep_clin(train)))
    Xc_te = sc.transform(imp.transform(prep_clin(test)))
    Xr_tr, Xr_te = pca_ft(train[rad].values, test[rad].values, 30)
    Xe_tr, Xe_te = pca_ft(train[emb].values, test[emb].values, 50)
    Xj_tr = np.hstack([Xc_tr, Xr_tr, Xe_tr])
    Xj_te = np.hstack([Xc_te, Xr_te, Xe_te])
    y_tr = train[['survival_time_days','event']].reset_index(drop=True)
    y_te = test[['survival_time_days','event']].reset_index(drop=True)
    return train, test, Xj_tr, Xj_te, y_tr, y_te

def fit_cox(X, y):
    d = pd.DataFrame(X, columns=[f'f{i}' for i in range(X.shape[1])])
    d['T'] = y['survival_time_days'].values
    d['E'] = y['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d, duration_col='T', event_col='E')
    return cph

def internal_cv(X, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cs = []
    for tr_idx, va_idx in kf.split(X):
        cph = fit_cox(X[tr_idx], y.iloc[tr_idx])
        h = cph.predict_partial_hazard(
            pd.DataFrame(X[va_idx], columns=[f'f{i}' for i in range(X.shape[1])])
        ).values
        cs.append(concordance_index(
            y.iloc[va_idx]['survival_time_days'].values, -h,
            y.iloc[va_idx]['event'].values))
    return float(np.mean(cs))

def external_c(X_tr, X_te, y_tr, y_te):
    cph = fit_cox(X_tr, y_tr)
    h = cph.predict_partial_hazard(
        pd.DataFrame(X_te, columns=[f'f{i}' for i in range(X_te.shape[1])])
    ).values
    return float(concordance_index(
        y_te['survival_time_days'].values, -h, y_te['event'].values))

print('===== 衰减分解表 =====')
rows = []
for tag, src, rp, ep in [('ComBat', 'combat', 'rad_combat_', 'emb_combat_'),
                          ('no ComBat', 'nocombat', 'original_', 'emb_')]:
    s, rad, emb = load(src, rp, ep)
    train, test, Xj_tr, Xj_te, y_tr, y_te = build(s, rad, emb)
    cv_c = internal_cv(Xj_tr, y_tr)
    ext_c = external_c(Xj_tr, Xj_te, y_tr, y_te)
    print(f'\n[{tag}]')
    print(f'  LUNG1 内部 5 折 CV C-index = {cv_c:.4f}')
    print(f'  Radiogenomics 外验 C-index = {ext_c:.4f}')
    print(f'  Δ (衰减)                    = {ext_c - cv_c:+.4f}')
    rows.append({'tag': tag, 'CV_C': round(cv_c,4),
                 'EXT_C': round(ext_c,4),
                 'delta': round(ext_c - cv_c, 4)})

pd.DataFrame(rows).to_csv(BASE + 'decay_decomp.csv', index=False)
print('\n已保存 decay_decomp.csv')
print('完成')
