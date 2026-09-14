import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/root/autodl-tmp/'

df = pd.read_csv(BASE + 'master_features_clean.csv',
                 encoding='utf-8-sig', low_memory=False)

surv = df[df['dataset'].isin(['LUNG1', 'Radiogenomics'])].copy()
surv = surv.dropna(subset=['survival_time_days', 'event']).copy()

rad_cols = [c for c in df.columns if c.startswith('original_')]
emb_cols = [c for c in df.columns if c.startswith('emb_')]
print('样本：', surv.shape, ' 影像组学列：', len(rad_cols), ' 嵌入列：', len(emb_cols))

# 影像组学：标准化 -> PCA2
imp_r = SimpleImputer(strategy='median')
sc_r  = StandardScaler()
Xr = sc_r.fit_transform(imp_r.fit_transform(surv[rad_cols]))
pca_r = PCA(n_components=2, random_state=0)
Xr_2d = pca_r.fit_transform(Xr)

is_l1 = (surv['dataset'] == 'LUNG1').values

plt.figure(figsize=(6,5))
plt.scatter(Xr_2d[is_l1,0],  Xr_2d[is_l1,1],  s=8, alpha=0.5, label='LUNG1')
plt.scatter(Xr_2d[~is_l1,0], Xr_2d[~is_l1,1], s=8, alpha=0.5, label='Radiogenomics')
plt.legend(); plt.title('Radiomics PCA2')
plt.xlabel('PC1 ({:.1f}%)'.format(pca_r.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 ({:.1f}%)'.format(pca_r.explained_variance_ratio_[1]*100))
plt.tight_layout()
plt.savefig(BASE + 'audit_radiomics_pca2.png', dpi=150)
print('已保存 audit_radiomics_pca2.png')

# 每个特征的标准化均值差异（Cohen's d）
mean_diffs = []
for i, c in enumerate(rad_cols):
    a = Xr[is_l1, i]
    b = Xr[~is_l1, i]
    d = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-8)
    mean_diffs.append((c, d, a.mean(), b.mean()))

mdf = pd.DataFrame(mean_diffs, columns=['feature','cohens_d','mean_LUNG1','mean_Radiogenomics'])
mdf['abs_d'] = mdf['cohens_d'].abs()
mdf = mdf.sort_values('abs_d', ascending=False)
print('\nCohen\'s d 最大的 10 个特征：')
print(mdf.head(10).to_string(index=False))
print('\nCohen\'s d 绝对值 > 0.8 的特征数：', (mdf['abs_d'] > 0.8).sum(), '/', len(mdf))
mdf.to_csv(BASE + 'audit_radiomics_cohens_d.csv', index=False)
print('已保存 audit_radiomics_cohens_d.csv')

# Deep 嵌入：标准化 -> PCA2
sc_e = StandardScaler()
Xe = sc_e.fit_transform(surv[emb_cols].values)
pca_e = PCA(n_components=2, random_state=0)
Xe_2d = pca_e.fit_transform(Xe)

plt.figure(figsize=(6,5))
plt.scatter(Xe_2d[is_l1,0],  Xe_2d[is_l1,1],  s=8, alpha=0.5, label='LUNG1')
plt.scatter(Xe_2d[~is_l1,0], Xe_2d[~is_l1,1], s=8, alpha=0.5, label='Radiogenomics')
plt.legend(); plt.title('Deep Embeddings PCA2')
plt.xlabel('PC1 ({:.1f}%)'.format(pca_e.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 ({:.1f}%)'.format(pca_e.explained_variance_ratio_[1]*100))
plt.tight_layout()
plt.savefig(BASE + 'audit_deep_pca2.png', dpi=150)
print('已保存 audit_deep_pca2.png')

# Deep 嵌入：每个特征的 Cohen's d
mean_diffs_e = []
for i in range(Xe.shape[1]):
    a = Xe[is_l1, i]
    b = Xe[~is_l1, i]
    d = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-8)
    mean_diffs_e.append(d)
mean_diffs_e = np.array(mean_diffs_e)
print('\nDeep 嵌入 Cohen\'s d 绝对值 > 0.8 的维度数：',
      (np.abs(mean_diffs_e) > 0.8).sum(), '/', Xe.shape[1])
print('Deep 嵌入 Cohen\'s d 绝对值均值：', round(np.abs(mean_diffs_e).mean(), 3))
print('影像组学 Cohen\'s d 绝对值均值：', round(mdf['abs_d'].mean(), 3))

print('\n完成')
