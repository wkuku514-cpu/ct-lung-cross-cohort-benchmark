# feature_stability_intervention.py
# 用 LUNG1 内部 CV 系数稳定性做特征筛选，验证能否提升跨队列外验
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

d = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
c = pd.read_csv(BASE + 'master_features_clean.csv',
                encoding='utf-8-sig', low_memory=False)
need = ['dataset','patient_id','age','gender','stage_clean','histology_clean']
d = d.merge(c[need].drop_duplicates(['dataset','patient_id']),
            on=['dataset','patient_id'], how='left')
s = d[d['dataset'].isin(['LUNG1','Radiogenomics'])].dropna(
    subset=['survival_time_days','event']).copy()
s['event'] = s['event'].astype(int)
train = s[s['dataset']=='LUNG1'].copy()
test  = s[s['dataset']=='Radiogenomics'].copy()

rad = [c for c in d.columns if c.startswith('rad_combat_')]
emb = [c for c in d.columns if c.startswith('emb_combat_')]

def prep_clin(dd):
    out = pd.DataFrame(index=dd.index)
    out['age'] = dd['age'].values
    out['gender_male'] = (dd['gender']=='male').astype(int).values
    for st in ['I','II','III']:
        out['stage_'+st] = (dd['stage_clean']==st).astype(int).values
    for h in ['adeno','squamous','large_cell','nos']:
        out['hist_'+h] = (dd['histology_clean']==h).astype(int).values
    return out

def pca_ft(tr, te, n, prefix):
    ix, sx = SimpleImputer(strategy='median'), StandardScaler()
    z_tr = sx.fit_transform(ix.fit_transform(tr))
    z_te = sx.transform(ix.transform(te))
    p = PCA(n_components=n, random_state=0)
    cols = [f'{prefix}{i}' for i in range(n)]
    return (pd.DataFrame(p.fit_transform(z_tr), columns=cols),
            pd.DataFrame(p.transform(z_te), columns=cols))

imp, sc = SimpleImputer(strategy='median'), StandardScaler()
Xc_tr = pd.DataFrame(sc.fit_transform(imp.fit_transform(prep_clin(train))),
                     columns=prep_clin(train).columns)
Xc_te = pd.DataFrame(sc.transform(imp.transform(prep_clin(test))),
                     columns=prep_clin(train).columns)
Xr_tr, Xr_te = pca_ft(train[rad].values, test[rad].values, 30, 'rad_pca_')
Xe_tr, Xe_te = pca_ft(train[emb].values, test[emb].values, 50, 'emb_pca_')
Xj_tr = pd.concat([Xc_tr, Xr_tr, Xe_tr], axis=1)
Xj_te = pd.concat([Xc_te, Xr_te, Xe_te], axis=1)
y_tr = train[['survival_time_days','event']].reset_index(drop=True)
y_te = test[['survival_time_days','event']].reset_index(drop=True)

# --- LUNG1 内部 5 折 CV 收集系数稳定性 ---
print('===== 步骤 1：LUNG1 内部 5 折 CV 收集系数 =====')
kf = KFold(n_splits=5, shuffle=True, random_state=42)
coefs = []
for tr_idx, _ in kf.split(Xj_tr):
    d_tr = Xj_tr.iloc[tr_idx].copy()
    d_tr['T'] = y_tr.iloc[tr_idx]['survival_time_days'].values
    d_tr['E'] = y_tr.iloc[tr_idx]['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d_tr, duration_col='T', event_col='E')
    coefs.append(cph.params_)
coef_df = pd.DataFrame(coefs)
cv_stab = (coef_df.std(0).abs() / (coef_df.mean(0).abs() + 1e-8))
print(f'89 分量 CV 稳定性中位数: {cv_stab.median():.3f}')

# --- 外验函数 ---
def ext_c(cols):
    if len(cols) < 3: return np.nan
    d_tr = Xj_tr[cols].copy()
    d_tr['T'] = y_tr['survival_time_days'].values
    d_tr['E'] = y_tr['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d_tr, duration_col='T', event_col='E')
    h = cph.predict_partial_hazard(Xj_te[cols]).values
    return concordance_index(y_te['survival_time_days'].values, -h,
                             y_te['event'].values)

print('\n===== 步骤 2：全量 vs 稳定子集 vs 随机子集 =====')
all_cols = list(Xj_tr.columns)
c_all = ext_c(all_cols)
print(f'全量模型 ({len(all_cols)} 分量): 外验 C = {c_all:.4f}')

rng = np.random.RandomState(0)
for thr in [0.15, 0.20, 0.25, 0.30]:
    keep = cv_stab[cv_stab < thr].index.tolist()
    keep = [k for k in keep if k in Xj_tr.columns]
    if len(keep) < 3:
        print(f'\n[thr={thr}] 分量太少({len(keep)})，跳过')
        continue
    c_stab = ext_c(keep)
    # 随机等量对照
    rand_cs = []
    for _ in range(20):
        rcols = list(rng.choice(all_cols, size=len(keep), replace=False))
        rand_cs.append(ext_c(rcols))
    rand_mean = np.mean(rand_cs)
    rand_std  = np.std(rand_cs)
    print(f'\n[thr={thr}] 保留 {len(keep)} 分量')
    print(f'  稳定子集外验 C = {c_stab:.4f}  (Δ vs 全量 = {c_stab - c_all:+.4f})')
    print(f'  随机同量对照 C = {rand_mean:.4f} ± {rand_std:.4f}')
    print(f'  稳定 vs 随机  = {c_stab - rand_mean:+.4f}')
    if len(keep) <= 15:
        print(f'  保留分量: {keep}')

print('\n完成')
