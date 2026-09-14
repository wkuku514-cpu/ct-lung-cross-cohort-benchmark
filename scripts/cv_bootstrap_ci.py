import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

df = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
clin = pd.read_csv(BASE + 'master_features_clean.csv',
                   encoding='utf-8-sig', low_memory=False)
need = ['dataset','patient_id','age','gender','stage_clean','histology_clean']
df = df.merge(clin[need].drop_duplicates(['dataset','patient_id']),
              on=['dataset','patient_id'], how='left')
l1 = df[df['dataset']=='LUNG1'].dropna(subset=['survival_time_days','event']).copy()
l1['event'] = l1['event'].astype(int)
rad_cols = [c for c in df.columns if c.startswith('rad_combat_')]
emb_cols = [c for c in df.columns if c.startswith('emb_combat_')]

def prep_clin(d):
    out = pd.DataFrame(index=d.index)
    out['age'] = d['age'].values
    out['gender_male'] = (d['gender']=='male').astype(int).values
    for s in ['I','II','III']:
        out['stage_'+s] = (d['stage_clean']==s).astype(int).values
    for h in ['adeno','squamous','large_cell','nos']:
        out['hist_'+h] = (d['histology_clean']==h).astype(int).values
    return out

def one_cv(seed):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    cs = []
    for tr_idx, va_idx in kf.split(l1):
        tr = l1.iloc[tr_idx].reset_index(drop=True)
        va = l1.iloc[va_idx].reset_index(drop=True)
        imp, sc = SimpleImputer(strategy='median'), StandardScaler()
        Xc_tr = sc.fit_transform(imp.fit_transform(prep_clin(tr)))
        Xc_va = sc.transform(imp.transform(prep_clin(va)))
        def pb(a, b, n):
            ix, sx = SimpleImputer(strategy='median'), StandardScaler()
            z_tr = sx.fit_transform(ix.fit_transform(a))
            z_va = sx.transform(ix.transform(b))
            p = PCA(n_components=n, random_state=0)
            return p.fit_transform(z_tr), p.transform(z_va)
        Xr_tr, Xr_va = pb(tr[rad_cols].values, va[rad_cols].values, 30)
        Xe_tr, Xe_va = pb(tr[emb_cols].values, va[emb_cols].values, 50)
        Xj_tr = np.hstack([Xc_tr, Xr_tr, Xe_tr])
        Xj_va = np.hstack([Xc_va, Xr_va, Xe_va])
        cols = [f'f{i}' for i in range(Xj_tr.shape[1])]
        d = pd.DataFrame(Xj_tr, columns=cols)
        d['T'] = tr['survival_time_days'].values
        d['E'] = tr['event'].values
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
        cph.fit(d, duration_col='T', event_col='E')
        h = cph.predict_partial_hazard(pd.DataFrame(Xj_va, columns=cols)).values
        cs.append(concordance_index(va['survival_time_days'].values, -h,
                                    va['event'].values))
    return np.mean(cs)

print('Bootstrap 50 seeds x 5-fold CV (Joint, ComBat):')
vals = []
import time
t0 = time.time()
for s in range(50):
    vals.append(one_cv(s))
    print(f'  seed {s+1}/50  t={time.time()-t0:.0f}s  mean={np.mean(vals):.4f}', flush=True)
vals = np.array(vals)
print(f'  mean = {vals.mean():.4f}')
print(f'  95% CI = [{np.percentile(vals,2.5):.4f}, {np.percentile(vals,97.5):.4f}]')
print('完成')
