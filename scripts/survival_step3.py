# ================================================================
# survival_step3.py
# 目的：用 ComBat 校正后特征重跑 LUNG1→Radiogenomics 外验
#       + LUNG1 内部 5 折 CV（sanity check）
# ================================================================
import pandas as pd
import numpy as np
import warnings, time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

# ---- 1. 读数据 ----
combat = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
print('surv_combat:', combat.shape)

clin = pd.read_csv(BASE + 'master_features_clean.csv',
                   encoding='utf-8-sig', low_memory=False)
print('master_features_clean:', clin.shape)
nonfeat = [c for c in clin.columns if not c.startswith(('original_', 'emb_'))]
print('master_features_clean 非特征列:', nonfeat)

clin_need = ['dataset', 'patient_id', 'age', 'gender', 'stage_clean', 'histology_clean']
miss = [c for c in clin_need if c not in clin.columns]
if miss:
    raise SystemExit('缺少临床列: ' + str(miss))

df = combat.merge(clin[clin_need].drop_duplicates(['dataset', 'patient_id']),
                  on=['dataset', 'patient_id'], how='left')
print('合并后:', df.shape, ' age 缺失:', df['age'].isna().sum())

# ---- 2. 切分 ----
surv = df[df['dataset'].isin(['LUNG1', 'Radiogenomics'])].copy()
surv = surv.dropna(subset=['survival_time_days', 'event']).copy()
surv['event'] = surv['event'].astype(int)
print('LUNG1:', (surv['dataset'] == 'LUNG1').sum(),
      ' Radiogenomics:', (surv['dataset'] == 'Radiogenomics').sum())

train = surv[surv['dataset'] == 'LUNG1'].copy()
test  = surv[surv['dataset'] == 'Radiogenomics'].copy()

rad_cols = [c for c in combat.columns if c.startswith('rad_combat_')]
emb_cols = [c for c in combat.columns if c.startswith('emb_combat_')]
print('rad_combat:', len(rad_cols), ' emb_combat:', len(emb_cols))

# ---- 3. 临床（与 step2 对齐）----
def prep_clin(d):
    out = pd.DataFrame(index=d.index)
    out['age'] = d['age'].values
    out['gender_male'] = (d['gender'] == 'male').astype(int).values
    for s in ['I', 'II', 'III']:
        out['stage_' + s] = (d['stage_clean'] == s).astype(int).values
    for h in ['adeno', 'squamous', 'large_cell', 'nos']:
        out['hist_' + h] = (d['histology_clean'] == h).astype(int).values
    return out

Xc_tr_raw = prep_clin(train)
Xc_te_raw = prep_clin(test)
imp_c = SimpleImputer(strategy='median')
sc_c  = StandardScaler()
Xc_tr = pd.DataFrame(sc_c.fit_transform(imp_c.fit_transform(Xc_tr_raw)),
                     columns=Xc_tr_raw.columns, index=train.index)
Xc_te = pd.DataFrame(sc_c.transform(imp_c.transform(Xc_te_raw)),
                     columns=Xc_tr_raw.columns, index=test.index)
print('临床特征:', Xc_tr.shape)

# ---- 4. 影像组学 PCA30 ----
imp_r = SimpleImputer(strategy='median')
sc_r  = StandardScaler()
Xr_tr_z = sc_r.fit_transform(imp_r.fit_transform(train[rad_cols]))
Xr_te_z = sc_r.transform(imp_r.transform(test[rad_cols]))
pca_r = PCA(n_components=30, random_state=0)
Xr_tr_pca = pd.DataFrame(pca_r.fit_transform(Xr_tr_z),
                         columns=['rad_pca_'+str(i) for i in range(30)],
                         index=train.index)
Xr_te_pca = pd.DataFrame(pca_r.transform(Xr_te_z),
                         columns=['rad_pca_'+str(i) for i in range(30)],
                         index=test.index)
print('影像组学 PCA30 解释方差:', round(pca_r.explained_variance_ratio_.sum(), 3))

# ---- 5. Deep PCA50 ----
imp_e = SimpleImputer(strategy='median')
sc_e  = StandardScaler()
Xe_tr_z = sc_e.fit_transform(imp_e.fit_transform(train[emb_cols]))
Xe_te_z = sc_e.transform(imp_e.transform(test[emb_cols]))
pca_e = PCA(n_components=50, random_state=0)
Xe_tr_pca = pd.DataFrame(pca_e.fit_transform(Xe_tr_z),
                         columns=['emb_pca_'+str(i) for i in range(50)],
                         index=train.index)
Xe_te_pca = pd.DataFrame(pca_e.transform(Xe_te_z),
                         columns=['emb_pca_'+str(i) for i in range(50)],
                         index=test.index)
print('Deep PCA50 解释方差:', round(pca_e.explained_variance_ratio_.sum(), 3))

# ---- 6. 联合 ----
Xj_tr = pd.concat([Xc_tr, Xr_tr_pca, Xe_tr_pca], axis=1)
Xj_te = pd.concat([Xc_te, Xr_te_pca, Xe_te_pca], axis=1)
print('联合特征:', Xj_tr.shape)

y_tr = train[['survival_time_days', 'event']].copy()
y_te = test[['survival_time_days', 'event']].copy()

# ---- 7. 评估函数 ----
def evaluate(name, X_tr, X_te, y_tr, y_te, penalizer=0.1):
    d_tr = X_tr.copy()
    d_tr['T'] = y_tr['survival_time_days'].values
    d_tr['E'] = y_tr['event'].values
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
    cph.fit(d_tr, duration_col='T', event_col='E')
    c_tr = cph.concordance_index_

    hazard_te = cph.predict_partial_hazard(X_te).values
    c_te = concordance_index(y_te['survival_time_days'].values,
                             -hazard_te, y_te['event'].values)

    times = [365, 730, 1095]
    aucs = np.array([np.nan]*3)
    try:
        y_surv_tr = Surv.from_arrays(y_tr['event'].astype(bool).values,
                                     y_tr['survival_time_days'].values)
        y_surv_te = Surv.from_arrays(y_te['event'].astype(bool).values,
                                     y_te['survival_time_days'].values)
        auc_vals, _ = cumulative_dynamic_auc(y_surv_tr, y_surv_te,
                                             hazard_te, times=times)
        aucs = auc_vals
    except Exception as e:
        print('  AUC 失败:', e)

    print(f'[{name}] train C={c_tr:.3f}  test C={c_te:.3f}  '
          f'AUC12/24/36m={aucs[0]:.3f}/{aucs[1]:.3f}/{aucs[2]:.3f}')
    return {'model': name, 'train_C': round(c_tr,3), 'test_C': round(c_te,3),
            'AUC_12m': round(aucs[0],3), 'AUC_24m': round(aucs[1],3),
            'AUC_36m': round(aucs[2],3)}

# ---- 8. 外验结果 ----
print('\n===== ComBat 后外验 =====')
results = []
results.append(evaluate('临床',        Xc_tr, Xc_te, y_tr, y_te))
results.append(evaluate('影像组学-PCA30', Xr_tr_pca, Xr_te_pca, y_tr, y_te))
results.append(evaluate('Deep-PCA50',   Xe_tr_pca, Xe_te_pca, y_tr, y_te))
results.append(evaluate('联合',        Xj_tr, Xj_te, y_tr, y_te))

res_df = pd.DataFrame(results)
res_df.to_csv(BASE + 'step3_results.csv', index=False)
print('\n已保存 step3_results.csv')
print(res_df.to_string(index=False))

# ---- 9. LUNG1 内部 5 折 CV ----
print('\n===== LUNG1 内部 5 折 CV（ComBat 后）=====')
kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_cindex(X, y, penalizer=0.1):
    cs = []
    for tr_idx, va_idx in kf.split(X):
        X_tr = X.iloc[tr_idx]; X_va = X.iloc[va_idx]
        y_tr_f = y.iloc[tr_idx]; y_va_f = y.iloc[va_idx]
        d = X_tr.copy()
        d['T'] = y_tr_f['survival_time_days'].values
        d['E'] = y_tr_f['event'].values
        try:
            cph = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
            cph.fit(d, duration_col='T', event_col='E')
            risk = -cph.predict_partial_hazard(X_va).values
            c = concordance_index(y_va_f['survival_time_days'].values,
                                  risk, y_va_f['event'].values)
            cs.append(c)
        except Exception as e:
            print('  fold 失败:', e)
    return float(np.mean(cs)), float(np.std(cs))

# 只取 LUNG1 行（train）——因为 CV 应在同一队列内做
y_l1 = y_tr.copy()
for nm, X in [('临床',        Xc_tr),
              ('影像组学-PCA30', Xr_tr_pca),
              ('Deep-PCA50',   Xe_tr_pca),
              ('联合',        Xj_tr)]:
    m, s = cv_cindex(X, y_l1)
    print(f'[{nm}] LUNG1-CV C={m:.3f} ± {s:.3f}')

print('\n完成')
