# longshort_external.py
# LUNG1 (survival) -> LUAD-CT-Survival (Long/Short 分类) 外验
import pandas as pd, numpy as np, warnings
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
warnings.filterwarnings('ignore')
BASE = '/root/autodl-tmp/'

df = pd.read_csv(BASE + 'master_features_clean.csv',
                 encoding='utf-8-sig', low_memory=False)
print('master_features_clean:', df.shape)

# --- 切分 ---
l1 = df[df['dataset'] == 'LUNG1'].dropna(
    subset=['survival_time_days', 'event']).copy()
l1['event'] = l1['event'].astype(int)
luad = df[df['dataset'] == 'LUAD-CT-Survival'].copy()

print('LUNG1 (train, survival):', l1.shape)
print('LUAD  (test, Long/Short):', luad.shape)
print('LUAD label 分布:')
print(luad['survival_label'].value_counts())

# --- 特征列 ---
rad_cols = [c for c in df.columns if c.startswith('original_')]
emb_cols = [c for c in df.columns if c.startswith('emb_')]
print('original_:', len(rad_cols), ' emb_:', len(emb_cols))

# --- PCA 流程（在 LUNG1 上 fit，在 LUAD 上 transform）---
def pca_fit_transform(tr, te, n, prefix):
    ix = SimpleImputer(strategy='median')
    sx = StandardScaler()
    z_tr = sx.fit_transform(ix.fit_transform(tr))
    z_te = sx.transform(ix.transform(te))
    p = PCA(n_components=n, random_state=0)
    cols = [f'{prefix}{i}' for i in range(n)]
    return (pd.DataFrame(p.fit_transform(z_tr), columns=cols, index=range(len(tr))),
            pd.DataFrame(p.transform(z_te),  columns=cols, index=range(len(te))),
            p.explained_variance_ratio_.sum())

Xr_tr, Xr_te, vr = pca_fit_transform(l1[rad_cols].values, luad[rad_cols].values, 30, 'r')
Xe_tr, Xe_te, ve = pca_fit_transform(l1[emb_cols].values, luad[emb_cols].values, 50, 'e')
Xj_tr = pd.concat([Xr_tr, Xe_tr], axis=1)
Xj_te = pd.concat([Xr_te, Xe_te], axis=1)
print(f'影像组学 PCA30 解释方差: {vr:.3f}')
print(f'Deep PCA50   解释方差: {ve:.3f}')
print('联合特征:', Xj_tr.shape)

# --- LUNG1 训练 Cox ---
def fit_cox(X_tr, y_tr):
    d = X_tr.copy()
    d['T'] = y_tr['survival_time_days'].values
    d['E'] = y_tr['event'].values
    cph = CoxPHFitter(penalizer=0.1, l1_ratio=1.0)
    cph.fit(d, duration_col='T', event_col='E')
    return cph

y_tr = l1[['survival_time_days', 'event']].reset_index(drop=True)

# --- 评估：LUAD 上 Long/Short 分类 ---
y_te = (luad['survival_label'] == 'Short').astype(int).values  # Short=1 (高风险)
print('\n===== LUAD 外验：Long/Short 分类 =====')
print(f'{"模型":<14}{"AUC":<10}{"Acc":<10}{"F1":<10}')
results = []
for name, X_tr, X_te in [('影像组学', Xr_tr, Xr_te),
                          ('Deep',     Xe_tr, Xe_te),
                          ('联合',     Xj_tr, Xj_te)]:
    try:
        cph = fit_cox(X_tr, y_tr)
        h = cph.predict_partial_hazard(X_te).values
        auc = roc_auc_score(y_te, h)
        # 阈值用 LUNG1 训练集 hazard 中位数
        h_tr = cph.predict_partial_hazard(X_tr).values
        thr = np.median(h_tr)
        pred = (h > thr).astype(int)
        acc = accuracy_score(y_te, pred)
        f1 = f1_score(y_te, pred, zero_division=0)
        print(f'{name:<14}{auc:<10.3f}{acc:<10.3f}{f1:<10.3f}')
        results.append({'model': name, 'AUC': round(auc,3),
                        'Acc': round(acc,3), 'F1': round(f1,3)})
    except Exception as e:
        print(f'{name:<14}FAILED: {e}')

pd.DataFrame(results).to_csv(BASE + 'longshort_results.csv', index=False)
print('\n已保存 longshort_results.csv')

# --- 基线：随机 ---
print('\n基线：')
print(f'  Long/Short 比例: {1-y_te.mean():.3f}/{y_te.mean():.3f}')
print(f'  随机 AUC = 0.500')
print('完成')
