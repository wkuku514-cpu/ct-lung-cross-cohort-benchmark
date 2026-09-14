import pandas as pd
import numpy as np

base = '/root/autodl-tmp/'
df = pd.read_csv(base + 'master_features.csv', encoding='utf-8-sig')

print('原始形状：', df.shape)

# ---------- 1. gender 统一 ----------
def clean_gender(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s in ['male', 'm']: return 'male'
    if s in ['female', 'f']: return 'female'
    return np.nan
df['gender'] = df['gender'].apply(clean_gender)
print('gender 分布：')
print(df.groupby('dataset')['gender'].value_counts(dropna=False))

# ---------- 2. histology 统一 ----------
def clean_histology(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    if s == '': return np.nan
    if 'squamous' in s: return 'squamous'
    if 'adeno' in s: return 'adeno'
    if 'large cell' in s: return 'large_cell'
    if 'nos' in s: return 'nos'
    return 'other'
df['histology_clean'] = df['histology'].apply(clean_histology)
print('\nhistology_clean 分布：')
print(df.groupby('dataset')['histology_clean'].value_counts(dropna=False))

# ---------- 3. stage 统一 ----------
def clean_stage(x):
    if pd.isna(x): return 'Unknown'
    s = str(x).strip()
    if s == '' or s.lower().startswith('not collected'):
        return 'Unknown'
    su = s.upper()
    # T/N/M 格式
    if su.startswith('T'):
        if 'M1' in su: return 'IV'
        if 'N3' in su: return 'III'
        if 'N2' in su: return 'III'
        if 'N1' in su: return 'II'
        if su.startswith('T4') or su.startswith('T3'): return 'II'
        return 'I'
    # III / II / I
    if su.startswith('III'): return 'III'
    if su.startswith('II'):  return 'II'
    if su.startswith('IV'):  return 'IV'
    if su.startswith('I'):   return 'I'
    # 数字
    if su.startswith('4'): return 'IV'
    if su.startswith('3'): return 'III'
    if su.startswith('2'): return 'II'
    if su.startswith('1'): return 'I'
    return 'Unknown'
df['stage_clean'] = df['stage'].apply(clean_stage)
print('\nstage_clean 分布：')
print(df.groupby('dataset')['stage_clean'].value_counts(dropna=False))

# ---------- 4. event 转 Int ----------
df['event'] = df['event'].astype('Int64')

# ---------- 5. age 数值化 ----------
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['age_missing'] = df['age'].isna().astype(int)
print('\nage 缺失数：', df['age_missing'].sum())

# ---------- 6. 保存 ----------
df.to_csv(base + 'master_features_clean.csv', index=False)
print('\n已保存到', base + 'master_features_clean.csv')
print('最终形状：', df.shape)
