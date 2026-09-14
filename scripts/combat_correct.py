import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from neuroCombat import neuroCombat

BASE = '/root/autodl-tmp/'

df = pd.read_csv(BASE + 'master_features_clean.csv',
                 encoding='utf-8-sig', low_memory=False)

surv = df[df['dataset'].isin(['LUNG1', 'Radiogenomics'])].copy()
surv = surv.dropna(subset=['survival_time_days', 'event']).copy()
surv = surv.reset_index(drop=True)

rad_cols = [c for c in df.columns if c.startswith('original_')]
emb_cols = [c for c in df.columns if c.startswith('emb_')]

imp_r = SimpleImputer(strategy='median')
R = imp_r.fit_transform(surv[rad_cols].values).astype(np.float32).T

imp_e = SimpleImputer(strategy='median')
E = imp_e.fit_transform(surv[emb_cols].values).astype(np.float32).T

batch = surv['dataset'].values
print('影像组学矩阵：', R.shape)
print('Deep 矩阵：', E.shape)
print('batch 分布：', pd.Series(batch).value_counts().to_dict())

print('ComBat 影像组学 ...')
R_corrected = neuroCombat(
    dat=R,
    covars=pd.DataFrame({'batch': batch}),
    batch_col='batch',
    categorical_cols=[],
    continuous_cols=[]
)['data']

print('ComBat Deep ...')
E_corrected = neuroCombat(
    dat=E,
    covars=pd.DataFrame({'batch': batch}),
    batch_col='batch',
    categorical_cols=[],
    continuous_cols=[]
)['data']

R_corrected = R_corrected.T
E_corrected = E_corrected.T
print('校正后影像组学：', R_corrected.shape)
print('校正后 Deep：', E_corrected.shape)

rad_df = pd.DataFrame(R_corrected,
                      columns=['rad_combat_' + c for c in rad_cols],
                      index=surv.index)
emb_df = pd.DataFrame(E_corrected,
                      columns=['emb_combat_' + c for c in emb_cols],
                      index=surv.index)

meta = surv[['dataset','patient_id','survival_time_days','event']].copy()
meta = pd.concat([meta, rad_df, emb_df], axis=1)
meta.to_csv(BASE + 'surv_combat.csv', index=False)
print('已保存 surv_combat.csv')
print('完成')
