# calibration_dca.py
# 目标：对 Deep-PCA50 和 联合（ComBat 后）做
#   1. Bootstrap 95% CI (test C)
#   2. 校准曲线 (12/24/36 个月, 4 分位分组)
#   3. 决策曲线分析 DCA
import pandas as pd
import numpy as np
import warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
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
surv = df[df['dataset'].isin(['LUNG1', 'Radiogenomics'])].dropna(
    subset=['survival_time_days', 'event']).copy()
surv['event'] = surv['event'].astype(int)
train = surv[surv['dataset'] == 'LUNG1'].copy()
test  = surv[surv['dataset'] == 'Radiogenomics'].copy()
print('train:', train.shape, ' test:', test.shape)

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

imp = SimpleImputer(strategy='median')
sc  = StandardScaler()
Xc_tr = pd.DataFrame(sc.fit_transform(imp.fit_transform(prep_clin(train))),
                     columns=prep_clin(train).columns, index=train.index)
Xc_te = pd.DataFrame(sc.transform(imp.transform(prep_clin(test))),
                     columns=prep_clin(train).columns, index=test.index)

def pca_block(tr_raw, te_raw, n, prefix):
    imp_x = SimpleImputer(strategy='median')
    sc_x  = StandardScaler()
    z_tr = sc_x.fit_transform(imp_x.fit_transform(tr_raw))
    z_te = sc_x.transform(imp_x.transform(te_raw))
    p = PCA(n_components=n, random_state=0)
    cols = [prefix + str(i) for i in range(n)]
    return (pd.DataFrame(p.fit_transform(z_tr), columns=cols, index=train.index),
            pd.DataFrame(p.transform(z_te), columns=cols, index=test.index))

Xr_tr, Xr_te = pca_block(train[rad_cols].values, test[rad_cols].values, 30, 'r')
Xe_tr, Xe_te = pca_block(train[emb_cols].values, test[emb_cols].values, 50, 'e')
Xj_tr = pd.concat([Xc_tr, Xr_tr, Xe_tr], axis=1)
Xj_te = pd.concat([Xc_te, Xr_te, Xe_te], axis=1)

y_tr = train[['survival_time_days', 'event']].copy()
y_te = test[['survival_time_days', 'event']].copy()

def fit_cox(X, y):
    d = X.copy()
    d['T'] = y['survival_time_days'].values
    d['E'] = y['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d, duration_col='T', event_col='E')
    return cph

cph_deep  = fit_cox(Xe_tr, y_tr)
cph_joint = fit_cox(Xj_tr, y_tr)

def bootstrap_c(cph, X_te, y_te, n_boot=500, seed=42):
    rng = np.random.RandomState(seed)
    h = cph.predict_partial_hazard(X_te).values
    t = y_te['survival_time_days'].values
    e = y_te['event'].values
    n = len(h); cs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        try:
            cs.append(concordance_index(t[idx], -h[idx], e[idx]))
        except Exception:
            pass
    return np.mean(cs), np.percentile(cs, 2.5), np.percentile(cs, 97.5)

print('\n===== Bootstrap 95% CI (test C) =====')
for nm, cph, X in [('Deep', cph_deep, Xe_te),
                   ('Joint', cph_joint, Xj_te)]:
    m, lo, hi = bootstrap_c(cph, X, y_te)
    print(f'[{nm}] {m:.3f} (95% CI {lo:.3f} - {hi:.3f})')

times = np.array([365, 730, 1095])

def calibration_plot(cph, X_te, y_te, name):
    h = cph.predict_partial_hazard(X_te).values
    surv_pred = cph.predict_survival_function(X_te, times=times).values.T
    order = np.argsort(-h)
    groups = np.array_split(order, 4)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ti, t in enumerate(times):
        ax = axes[ti]
        preds, obs = [], []
        for g in groups:
            pred = surv_pred[g, ti].mean()
            kmf = KaplanMeierFitter()
            kmf.fit(y_te['survival_time_days'].values[g],
                    y_te['event'].values[g])
            obs.append(kmf.predict(t))
            preds.append(pred)
        ax.scatter(preds, obs, s=80, c='steelblue')
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('Predicted S(t)')
        ax.set_ylabel('Observed S(t) (KM)')
        ax.set_title(f'{name} @ {t}d')
        ax.set_xlim(0.1, 1); ax.set_ylim(0.1, 1)
    plt.tight_layout()
    plt.savefig(BASE + f'calib_{name}.png', dpi=150)
    plt.close()
    print(f'已保存 calib_{name}.png')

calibration_plot(cph_deep,  Xe_te, y_te, 'Deep')
calibration_plot(cph_joint, Xj_te, y_te, 'Joint')

def dca(cph, X_te, y_te, t, thresholds):
    h = cph.predict_partial_hazard(X_te).values
    h_norm = (h - h.min()) / (h.max() - h.min() + 1e-8)
    km_cens = KaplanMeierFitter()
    km_cens.fit(y_te['survival_time_days'], 1 - y_te['event'])
    G = lambda tt: km_cens.predict(tt)
    te_time = y_te['survival_time_days'].values
    te_evt  = y_te['event'].values
    n = len(te_time); nb = []
    for pt in thresholds:
        pred_pos = h_norm >= pt
        event_by_t = (te_time <= t) & (te_evt == 1)
        w = np.where(event_by_t, 1.0 / np.maximum(G(te_time), 1e-6), 0)
        TP = np.sum(w * pred_pos) / n
        alive_at_t = te_time > t
        FP = np.sum((pred_pos & alive_at_t).astype(float)) / n
        nb.append(TP - FP * (pt / (1 - pt)))
    return np.array(nb)

thresholds = np.arange(0.05, 0.95, 0.05)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ti, t in enumerate(times):
    ax = axes[ti]
    for nm, cph, X in [('Deep', cph_deep, Xe_te),
                       ('Joint', cph_joint, Xj_te)]:
        nb = dca(cph, X, y_te, t, thresholds)
        ax.plot(thresholds, nb, label=nm)
    kmf = KaplanMeierFitter()
    kmf.fit(y_te['survival_time_days'], y_te['event'])
    prev = 1 - kmf.predict(t)
    treat_all = prev - (1 - prev) * (thresholds / (1 - thresholds))
    ax.plot(thresholds, treat_all, 'k--', label='Treat all')
    ax.axhline(0, color='gray', lw=1, label='Treat none')
    ax.set_xlabel('Threshold probability')
    ax.set_ylabel('Net benefit')
    ax.set_title(f'DCA @ {t}d')
    ax.set_ylim(-0.2, max(0.5, prev + 0.1))
    ax.legend()
plt.tight_layout()
plt.savefig(BASE + 'dca.png', dpi=150)
plt.close()
print('已保存 dca.png')
print('完成')
