import pandas as pd

base = '/root/autodl-tmp/'

man  = pd.read_csv(base + 'manifest.csv', encoding='utf-8-sig')
clin = pd.read_csv(base + 'master_clinical.csv', encoding='utf-8-sig')
rad  = pd.read_csv(base + 'preprocessed/features/radiomics_features.csv', encoding='utf-8-sig')
deep = pd.read_csv(base + 'preprocessed/features/deep_embeddings.csv', encoding='utf-8-sig')

# 统一 key
man['key']  = man['dataset'].astype(str) + '_' + man['patient_id'].astype(str)
clin['key'] = clin['dataset'].astype(str) + '_' + clin['patient_id'].astype(str)
rad  = rad.rename(columns={'patient_id': 'key'})
deep = deep.rename(columns={'patient_id': 'key'})

# 先以 manifest 为主表
df = man[['key', 'dataset', 'patient_id', 'study_uid', 'series_uid', 'n_files', 'folder']].copy()

# 合并影像组学
df = df.merge(rad, on='key', how='left')
print('合并 radiomics 后：', df.shape)

# 合并深度学习嵌入
df = df.merge(deep, on='key', how='left')
print('合并 deep 后：', df.shape)

# 合并临床（去掉 dataset/patient_id 避免列冲突）
clin_small = clin.drop(columns=['dataset', 'patient_id'])
df = df.merge(clin_small, on='key', how='left')
print('合并 clinical 后：', df.shape)

# 检查
print('\n列数：', df.shape[1])
print('行数：', df.shape[0])
print('key 重复：', df['key'].duplicated().sum())
print('临床缺失行数（应为 47，即 QIN-LUNG-CT）：', df['survival_time_days'].isna().sum())

# 保存
df.to_csv(base + 'master_features.csv', index=False)
print('\n已保存到', base + 'master_features.csv')
