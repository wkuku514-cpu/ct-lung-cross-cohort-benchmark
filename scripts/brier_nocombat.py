# brier_nocombat.py — ComBat 前的 Brier 对照
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
from sksurv.metrics import brier_score
from sksurv.util import Surv
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

df = pd.read_csv(BASE + 'master_features_clean.csv',
                 encoding='utf-8-sig', low_memory=False)
surv = df[df['dataset'].isin(['LUNG1','Radiogenomics'])].dropna(
    subset=['survival_time_days','event']).copy()
surv['event'] = surv['event'].astype(int)
train = surv[surv['dataset']=='LUNG1'].copy()
test  = surv[surv['dataset']=='Radiogenomics'].copy()

rad_cols = [c for c in df.columns if c.startswith('original_')]
emb_cols = [c for c in df.columns if c.startswith('emb_')]
print('original_:', len(rad_cols), ' emb_:', len(emb_cols))

def prep_clin(d):
    out = pd.DataFrame(index=d.index)
    out['age'] = d['age'].values
    out['gender_male'] = (d['gender']=='male').astype(int).values
    for s in ['I','II','III']:
        out['stage_'+s] = (d['stage_clean']==s).astype(int).values
    for h in ['adeno','squamous','large_cell','nos']:
        out['hist_'+h] = (d['histology_clean']==h).astype(int).values
    return out

imp, sc = SimpleImputer(strategy='median'), StandardScaler()
Xc_tr = sc.fit_transform(imp.fit_transform(prep_clin(train)))
Xc_te = sc.transform(imp.transform(prep_clin(test)))
def pca_block(tr, te, n):
    imp_x, sc_x = SimpleImputer(strategy='median'), StandardScaler()
    z_tr = sc_x.fit_transform(imp_x.fit_transform(tr))
    z_te = sc_x.transform(imp_x.transform(te))
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(z_tr), p.transform(z_te)
Xr_tr, Xr_te = pca_block(train[rad_cols].values, test[rad_cols].values, 30)
Xe_tr, Xe_te = pca_block(train[emb_cols].values, test[emb_cols].values, 50)
Xj_tr = np.hstack([Xc_tr, Xr_tr, Xe_tr])
Xj_te = np.hstack([Xc_te, Xr_te, Xe_te])
cols = [f'f{i}' for i in range(Xj_tr.shape[1])]
d_tr = pd.DataFrame(Xj_tr, columns=cols)
d_tr['T'] = train['survival_time_days'].values
d_tr['E'] = train['event'].values
cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
cph.fit(d_tr, duration_col='T', event_col='E')

times = np.array([365,730,1095])
t_te, e_te = test['survival_time_days'].values, test['event'].values
y_te_surv = Surv.from_arrays(e_te.astype(bool), t_te)
y_tr_surv = Surv.from_arrays(train['event'].astype(bool).values,
                             train['survival_time_days'].values)

s_orig = cph.predict_survival_function(pd.DataFrame(Xj_te, columns=cols),
                                       times=times).values.T
eta_te = cph.predict_log_partial_hazard(pd.DataFrame(Xj_te, columns=cols)).values
d_cal = pd.DataFrame({'eta': eta_te, 'T': t_te, 'E': e_te})
cph_cal = CoxPHFitter(); cph_cal.fit(d_cal, duration_col='T', event_col='E')
print('校准斜率 =', round(cph_cal.params_['eta'], 4))
s_cal = cph_cal.predict_survival_function(pd.DataFrame({'eta': eta_te}),
                                          times=times).values.T

print('\n===== Brier Score (no ComBat, IPCW from LUNG1) =====')
for ti, t in enumerate(times):
    bs_o = float(brier_score(y_tr_surv, y_te_surv, s_orig[:, ti], t)[1][0])
    bs_c = float(brier_score(y_tr_surv, y_te_surv, s_cal[:, ti], t)[1][0])
    print(f'  t={t}d  Original={bs_o:.4f}  Recalibrated={bs_c:.4f}  '
          f'Δ={bs_c-bs_o:+.4f}')
print('完成')
