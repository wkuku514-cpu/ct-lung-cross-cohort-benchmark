# recalibration.py
# 目标：证明跨队列失效在 calibration 层，不在 discrimination 层
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

combat = pd.read_csv(BASE + 'master_features_clean.csv', encoding='utf-8-sig', low_memory=False)
clin = combat
clin_need = ['dataset','patient_id','age','gender','stage_clean','histology_clean']
df = combat.copy()
surv = df[df['dataset'].isin(['LUNG1','Radiogenomics'])].dropna(
    subset=['survival_time_days','event']).copy()
surv['event'] = surv['event'].astype(int)
train = surv[surv['dataset']=='LUNG1'].copy()
test  = surv[surv['dataset']=='Radiogenomics'].copy()
rad_cols = [c for c in combat.columns if c.startswith('original_')]
emb_cols = [c for c in combat.columns if c.startswith('emb_')]

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
eta_tr = cph.predict_log_partial_hazard(pd.DataFrame(Xj_tr, columns=cols)).values
eta_te = cph.predict_log_partial_hazard(pd.DataFrame(Xj_te, columns=cols)).values

times = np.array([365,730,1095])
t_te = test['survival_time_days'].values
e_te = test['event'].values

# 原始 C-index（rank-preserving 变换不变）
c_orig = concordance_index(t_te, -eta_te, e_te)

# recalibration：在 test 上把 eta 作为单变量拟合 Cox
d_cal = pd.DataFrame({'eta': eta_te, 'T': t_te, 'E': e_te})
cph_cal = CoxPHFitter()
cph_cal.fit(d_cal, duration_col='T', event_col='E')
print(f'校准斜率 (eta coefficient): {cph_cal.params_["eta"]:.3f}')
print(f'原始 C-index = {c_orig:.3f}')

# 用校准后模型生成新的生存预测
X_eta = pd.DataFrame({'eta': eta_te})
s_cal = cph_cal.predict_survival_function(X_eta, times=times).values.T

# 用原始 Cox 的基线生成未校准预测，便于并排
s_orig = cph.predict_survival_function(pd.DataFrame(Xj_te, columns=cols),
                                       times=times).values.T

# 校准曲线对照：原 vs 校准后
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
km_cache = {}
def obs_at(groups, t):
    out = []
    for g in groups:
        kmf = KaplanMeierFitter()
        kmf.fit(t_te[g], e_te[g])
        out.append(kmf.predict(t))
    return out

for ti, t in enumerate(times):
    order = np.argsort(-eta_te)
    groups = np.array_split(order, 4)
    obs = obs_at(groups, t)
    # 上排：原始
    preds_o = [s_orig[g, ti].mean() for g in groups]
    axes[0, ti].scatter(preds_o, obs, s=80, c='steelblue')
    axes[0, ti].plot([0,1],[0,1],'k--', lw=1)
    axes[0, ti].set_title(f'Original @ {t}d')
    axes[0, ti].set_xlim(0.1,1); axes[0, ti].set_ylim(0.1,1)
    axes[0, ti].set_xlabel('Predicted S(t)'); axes[0, ti].set_ylabel('Observed S(t)')
    # 下排：recalibrated
    preds_c = [s_cal[g, ti].mean() for g in groups]
    axes[1, ti].scatter(preds_c, obs, s=80, c='darkorange')
    axes[1, ti].plot([0,1],[0,1],'k--', lw=1)
    axes[1, ti].set_title(f'Recalibrated @ {t}d')
    axes[1, ti].set_xlim(0.1,1); axes[1, ti].set_ylim(0.1,1)
    axes[1, ti].set_xlabel('Predicted S(t)'); axes[1, ti].set_ylabel('Observed S(t)')
plt.tight_layout()
plt.savefig(BASE + 'calib_recalibrated_nocombat.png', dpi=150)
plt.close()
print('已保存 calib_recalibrated_nocombat.png')

# DCA：原 vs 校准后
thresholds = np.arange(0.05, 0.95, 0.05)
def dca_from_surv(s_pred, t):
    km_cens = KaplanMeierFitter().fit(t_te, 1 - e_te)
    G = lambda tt: km_cens.predict(tt)
    n = len(t_te); nb = []
    for ti_pt in thresholds:
        pred_pos = s_pred < (1 - ti_pt)  # S(t) 低 -> 预测事件概率高
        event_by_t = (t_te <= t) & (e_te == 1)
        w = np.where(event_by_t, 1.0/np.maximum(G(t_te),1e-6), 0)
        TP = np.sum(w * pred_pos) / n
        alive = t_te > t
        FP = np.sum((pred_pos & alive).astype(float)) / n
        nb.append(TP - FP * (ti_pt/(1-ti_pt)))
    return np.array(nb)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ti, t in enumerate(times):
    ax = axes[ti]
    ax.plot(thresholds, dca_from_surv(s_orig[:, ti], t), label='Original')
    ax.plot(thresholds, dca_from_surv(s_cal[:, ti], t), label='Recalibrated')
    kmf = KaplanMeierFitter().fit(t_te, e_te)
    prev = 1 - kmf.predict(t)
    ax.plot(thresholds, prev - (1-prev)*(thresholds/(1-thresholds)),
            'k--', label='Treat all')
    ax.axhline(0, color='gray', lw=1, label='Treat none')
    ax.set_xlabel('Threshold probability'); ax.set_ylabel('Net benefit')
    ax.set_title(f'DCA @ {t}d'); ax.legend()
    ax.set_ylim(-0.2, max(0.5, prev + 0.1))
plt.tight_layout()
plt.savefig(BASE + 'dca_recalibrated_nocombat.png', dpi=150)
plt.close()
print('已保存 dca_recalibrated_nocombat.png')
print('完成')
