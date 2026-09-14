# shap_stability.py — Cox 系数稳定性 + LUNG1->Radiogenomics 排序一致性
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from scipy.stats import spearmanr
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

def prep_clin(d):
    out = pd.DataFrame(index=d.index)
    out['age'] = d['age'].values
    out['gender_male'] = (d['gender']=='male').astype(int).values
    for s in ['I','II','III']:
        out['stage_'+s] = (d['stage_clean']==s).astype(int).values
    for h in ['adeno','squamous','large_cell','nos']:
        out['hist_'+h] = (d['histology_clean']==h).astype(int).values
    return out

def pca_ft(tr, te, n, prefix):
    ix, sx = SimpleImputer(strategy='median'), StandardScaler()
    z_tr = sx.fit_transform(ix.fit_transform(tr))
    z_te = sx.transform(ix.transform(te))
    p = PCA(n_components=n, random_state=0)
    cols = [f'{prefix}{i}' for i in range(n)]
    return (pd.DataFrame(p.fit_transform(z_tr), columns=cols, index=range(len(tr))),
            pd.DataFrame(p.transform(z_te),  columns=cols, index=range(len(te))))

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

# --- LUNG1 内部 5 折拟合，收集系数 ---
print('===== Cox 系数稳定性（LUNG1 内部 5 折，ComBat 后联合模型）=====')
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
mean_coef = coef_df.mean(0)
std_coef  = coef_df.std(0)
cv_coef   = (std_coef.abs() / (mean_coef.abs() + 1e-8)).sort_values()
print('\n最稳定的前 15 个分量（5 折 CV 最小）:')
print(cv_coef.head(15).to_string())
print(f'\n系数 CV 中位数: {cv_coef.median():.3f}')

# --- LUNG1 全量拟合 vs Radiogenomics 拟合的排序一致性 ---
print('\n===== LUNG1 全量 Cox 系数 vs Radiogenomics 全量 Cox 系数 =====')
d_l1 = Xj_tr.copy()
d_l1['T'] = y_tr['survival_time_days'].values
d_l1['E'] = y_tr['event'].values
cph_l1 = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
cph_l1.fit(d_l1, duration_col='T', event_col='E')

d_rg = Xj_te.copy()
d_rg['T'] = y_te['survival_time_days'].values
d_rg['E'] = y_te['event'].values
cph_rg = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
cph_rg.fit(d_rg, duration_col='T', event_col='E')

common = [c for c in cph_l1.params_.index if c in cph_rg.params_.index]
rho, pval = spearmanr(cph_l1.params_[common], cph_rg.params_[common])
print(f'共同分量数: {len(common)}')
print(f'Spearman 相关: rho = {rho:.4f}, p = {pval:.4f}')

# top-K 重叠（按 |coef| 排序）
for K in [5, 10, 20]:
    top_l1 = cph_l1.params_[common].abs().nlargest(K).index
    top_rg = cph_rg.params_[common].abs().nlargest(K).index
    overlap = len(set(top_l1) & set(top_rg))
    print(f'  Top-{K} 重叠: {overlap}/{K} = {overlap/K:.2%}')

# 保存
out = pd.DataFrame({
    'component': common,
    'coef_LUNG1': cph_l1.params_[common].values,
    'coef_Radiogenomics': cph_rg.params_[common].values,
    'cv_stability': cv_coef.reindex(common).values
})
out.to_csv(BASE + 'coef_stability.csv', index=False)
print('\n已保存 coef_stability.csv')
print('完成')
