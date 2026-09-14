# calib_inside_lung1.py
# LUNG1 内部 5 折 CV 的校准曲线 + DCA，作为"跨队列校准偏移"的对照
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

combat = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
clin = pd.read_csv(BASE + 'master_features_clean.csv',
                   encoding='utf-8-sig', low_memory=False)
clin_need = ['dataset', 'patient_id', 'age', 'gender',
             'stage_clean', 'histology_clean']
df = combat.merge(clin[clin_need].drop_duplicates(['dataset', 'patient_id']),
                  on=['dataset', 'patient_id'], how='left')
l1 = df[df['dataset'] == 'LUNG1'].dropna(
    subset=['survival_time_days', 'event']).copy()
l1['event'] = l1['event'].astype(int)

rad_cols = [c for c in combat.columns if c.startswith('rad_combat_')]
emb_cols = [c for c in combat.columns if c.startswith('emb_combat_')]

def prep_clin(d):
    out = pd.DataFrame(index=d.index)
    out['age'] = d['age'].values
    out['gender_male'] = (d['gender'] == 'male').astype(int).values
    for s in ['I', 'II', 'III']:
        out['stage_' + s] = (d['stage_clean'] == s).astype(int).values
    for h in ['adeno', 'squamous', 'large_cell', 'nos']:
        out['hist_' + h] = (d['histology_clean'] == h).astype(int).values
    return out

times = np.array([365, 730, 1095])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 对每个 fold 拟合，收集 out-of-fold 预测
oof_pred = {t: [] for t in times}
oof_obs  = {t: [] for t in times}
oof_t    = []
oof_e    = []
oof_h    = np.zeros(len(l1))

def pca_block_fit(tr_raw, te_raw, n):
    imp_x = SimpleImputer(strategy='median')
    sc_x  = StandardScaler()
    z_tr = sc_x.fit_transform(imp_x.fit_transform(tr_raw))
    z_te = sc_x.transform(imp_x.transform(te_raw))
    p = PCA(n_components=n, random_state=0)
    return p.fit_transform(z_tr), p.transform(z_te)

fold_i = 0
for tr_idx, va_idx in kf.split(l1):
    fold_i += 1
    tr = l1.iloc[tr_idx].reset_index(drop=True)
    va = l1.iloc[va_idx].reset_index(drop=True)

    imp_c = SimpleImputer(strategy='median')
    sc_c  = StandardScaler()
    Xc_tr = sc_c.fit_transform(imp_c.fit_transform(prep_clin(tr)))
    Xc_va = sc_c.transform(imp_c.transform(prep_clin(va)))

    Xr_tr, Xr_va = pca_block_fit(tr[rad_cols].values, va[rad_cols].values, 30)
    Xe_tr, Xe_va = pca_block_fit(tr[emb_cols].values, va[emb_cols].values, 50)

    Xj_tr = np.hstack([Xc_tr, Xr_tr, Xe_tr])
    Xj_va = np.hstack([Xc_va, Xr_va, Xe_va])

    d_tr = pd.DataFrame(Xj_tr,
                        columns=[f'f{i}' for i in range(Xj_tr.shape[1])])
    d_tr['T'] = tr['survival_time_days'].values
    d_tr['E'] = tr['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d_tr, duration_col='T', event_col='E')

    Xj_va_df = pd.DataFrame(Xj_va,
                            columns=[f'f{i}' for i in range(Xj_va.shape[1])])
    h_va = cph.predict_partial_hazard(Xj_va_df).values
    oof_h[va_idx] = h_va
    oof_t.extend(va['survival_time_days'].values)
    oof_e.extend(va['event'].values)

    s_va = cph.predict_survival_function(Xj_va_df, times=times).values.T
    for ti, t in enumerate(times):
        oof_pred[t].extend(s_va[:, ti])
        oof_obs[t].extend([np.nan]*len(va))
    print(f'  fold {fold_i} 完成')

oof_t = np.array(oof_t); oof_e = np.array(oof_e)
for ti, t in enumerate(times):
    oof_pred[t] = np.array(oof_pred[t])

# 校准曲线（LUNG1 内部 OOF）
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ti, t in enumerate(times):
    ax = axes[ti]
    order = np.argsort(-oof_h)
    groups = np.array_split(order, 4)
    preds, obs = [], []
    for g in groups:
        pred = oof_pred[t][g].mean()
        kmf = KaplanMeierFitter()
        kmf.fit(oof_t[g], oof_e[g])
        obs.append(kmf.predict(t))
        preds.append(pred)
    ax.scatter(preds, obs, s=80, c='darkorange')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('Predicted S(t)')
    ax.set_ylabel('Observed S(t) (KM)')
    ax.set_title(f'LUNG1 OOF @ {t}d')
    ax.set_xlim(0.1, 1); ax.set_ylim(0.1, 1)
plt.tight_layout()
plt.savefig(BASE + 'calib_inside_lung1.png', dpi=150)
plt.close()
print('已保存 calib_inside_lung1.png')

# DCA（LUNG1 内部 OOF）
def dca_oof(h, t_arr, e_arr, t, thresholds):
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
    km_cens = KaplanMeierFitter().fit(t_arr, 1 - e_arr)
    G = lambda tt: km_cens.predict(tt)
    n = len(t_arr); nb = []
    for pt in thresholds:
        pred_pos = h_norm >= pt
        event_by_t = (t_arr <= t) & (e_arr == 1)
        w = np.where(event_by_t, 1.0 / np.maximum(G(t_arr), 1e-6), 0)
        TP = np.sum(w * pred_pos) / n
        alive_at_t = t_arr > t
        FP = np.sum((pred_pos & alive_at_t).astype(float)) / n
        nb.append(TP - FP * (pt / (1 - pt)))
    return np.array(nb)

thresholds = np.arange(0.05, 0.95, 0.05)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ti, t in enumerate(times):
    ax = axes[ti]
    nb = dca_oof(oof_h, oof_t, oof_e, t, thresholds)
    ax.plot(thresholds, nb, label='Joint (LUNG1 OOF)')
    kmf = KaplanMeierFitter().fit(oof_t, oof_e)
    prev = 1 - kmf.predict(t)
    treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, treat_all, 'k--', label='Treat all')
    ax.axhline(0, color='gray', lw=1, label='Treat none')
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'DCA inside LUNG1 @ {t}d')
    ax.set_ylim(-0.2, max(0.5, prev + 0.1))
    ax.legend()
plt.tight_layout()
plt.savefig(BASE + 'dca_inside_lung1.png', dpi=150)
plt.close()
print('已保存 dca_inside_lung1.png')
print('完成')
