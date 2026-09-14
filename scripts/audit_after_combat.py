import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/root/autodl-tmp/'

combat = pd.read_csv(BASE + 'surv_combat.csv', encoding='utf-8-sig')
print('surv_combat：', combat.shape)

rad_cols = [c for c in combat.columns if c.startswith('rad_combat_')]
emb_cols = [c for c in combat.columns if c.startswith('emb_combat_')]
print('校正后影像组学列数：', len(rad_cols))
print('校正后 Deep 列数：', len(emb_cols))

is_l1 = (combat['dataset'] == 'LUNG1').values

def cohens_d(X):
    a = X[is_l1]; b = X[~is_l1]
    m = (a.mean(0) - b.mean(0))
    s = np.sqrt((a.var(0) + b.var(0)) / 2) + 1e-8
    return m / s

# 影像组学
sc = StandardScaler()
R = sc.fit_transform(combat[rad_cols].values)
d_r = np.abs(cohens_d(R))
print('\n[ComBat 后] 影像组学 Cohen\'s d > 0.8 特征数：', (d_r > 0.8).sum(), '/', len(d_r))
print('[ComBat 后] 影像组学 Cohen\'s d 均值：', round(d_r.mean(), 3))

pca = PCA(n_components=2, random_state=0)
R2 = pca.fit_transform(R)
plt.figure(figsize=(6,5))
plt.scatter(R2[is_l1,0],  R2[is_l1,1],  s=8, alpha=0.5, label='LUNG1')
plt.scatter(R2[~is_l1,0], R2[~is_l1,1], s=8, alpha=0.5, label='Radiogenomics')
plt.legend(); plt.title('Radiomics PCA2 (after ComBat)')
plt.xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
plt.tight_layout(); plt.savefig(BASE + 'audit_radiomics_pca2_after.png', dpi=150)
print('已保存 audit_radiomics_pca2_after.png')

# Deep
sc = StandardScaler()
E = sc.fit_transform(combat[emb_cols].values)
d_e = np.abs(cohens_d(E))
print('\n[ComBat 后] Deep Cohen\'s d > 0.8 维度数：', (d_e > 0.8).sum(), '/', len(d_e))
print('[ComBat 后] Deep Cohen\'s d 均值：', round(d_e.mean(), 3))

pca = PCA(n_components=2, random_state=0)
E2 = pca.fit_transform(E)
plt.figure(figsize=(6,5))
plt.scatter(E2[is_l1,0],  E2[is_l1,1],  s=8, alpha=0.5, label='LUNG1')
plt.scatter(E2[~is_l1,0], E2[~is_l1,1], s=8, alpha=0.5, label='Radiogenomics')
plt.legend(); plt.title('Deep Embeddings PCA2 (after ComBat)')
plt.xlabel('PC1 ({:.1f}%)'.format(pca.explained_variance_ratio_[0]*100))
plt.ylabel('PC2 ({:.1f}%)'.format(pca.explained_variance_ratio_[1]*100))
plt.tight_layout(); plt.savefig(BASE + 'audit_deep_pca2_after.png', dpi=150)
print('已保存 audit_deep_pca2_after.png')
print('完成')
