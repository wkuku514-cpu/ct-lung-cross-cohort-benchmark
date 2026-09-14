import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

combat = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
clin = pd.read_csv(BASE + 'master_features_clean.csv',
                   encoding='utf-8-sig', low_memory=False)
clin_need = ['dataset','patient_id','age','gender','stage_clean','histology_clean']
df = combat.merge(clin[clin_need].drop_duplicates(['dataset','patient_id']),
                  on=['dataset','patient_id'], how='left')
surv = df[df['dataset'].isin(['LUNG1','Radiogenomics'])].dropna(
    subset=['survival_time_days','event']).copy()
surv['event'] = surv['event'].astype(int)
train = surv[surv['dataset']=='LUNG1'].copy()
test  = surv[surv['dataset']=='Radiogenomics'].copy()
rad_cols = [c for c in combat.columns if c.startswith('rad_combat_')]
emb_cols = [c for c in combat.columns if c.startswith('emb_combat_')]

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

s_orig = cph.predict_survival_function(pd.DataFrame(Xj_te, columns=cols),
                                       times=times).values.T
eta_te = cph.predict_log_partial_hazard(pd.DataFrame(Xj_te, columns=cols)).values
d_cal = pd.DataFrame({'eta': eta_te, 'T': t_te, 'E': e_te})
cph_cal = CoxPHFitter(); cph_cal.fit(d_cal, duration_col='T', event_col='E')
print('校准斜率 =', round(cph_cal.params_['eta'], 4))
s_cal = cph_cal.predict_survival_function(pd.DataFrame({'eta': eta_te}),
                                          times=times).values.T

print('\n=== s_orig (S(t)) 分布 ===')
for ti, t in enumerate(times):
    v = s_orig[:, ti]
    print(f'  t={t}: min={v.min():.4f}  q25={np.percentile(v,25):.4f}  '
          f'median={np.median(v):.4f}  q75={np.percentile(v,75):.4f}  max={v.max():.4f}')

print('\n=== s_cal (S(t)) 分布 ===')
for ti, t in enumerate(times):
    v = s_cal[:, ti]
    print(f'  t={t}: min={v.min():.4f}  q25={np.percentile(v,25):.4f}  '
          f'median={np.median(v):.4f}  q75={np.percentile(v,75):.4f}  max={v.max():.4f}')

print('\n=== 真实事件率 vs 预测事件率均值 ===')
for ti, t in enumerate(times):
    evt_frac = ((t_te <= t) & (e_te == 1)).mean()
    print(f'  t={t}: 真实事件率={evt_frac:.4f}  '
          f'orig预测={1-s_orig[:, ti].mean():.4f}  '
          f'cal预测={1-s_cal[:, ti].mean():.4f}')
print('完成')
