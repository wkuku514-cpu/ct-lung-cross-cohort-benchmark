import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from radiomics import featureextractor
import logging
import time
import sys

logging.getLogger('radiomics').setLevel(logging.ERROR)

img_dir = Path('/root/autodl-tmp/preprocessed/images')
mask_dir = Path('/root/autodl-tmp/preprocessed/lung_masks')
out_dir = Path('/root/autodl-tmp/preprocessed/features')
out_dir.mkdir(parents=True, exist_ok=True)

out_csv = out_dir / 'radiomics_features.csv'
err_log = out_dir / 'radiomics_errors.csv'

params = {
    'binWidth': 25,
    'resampledPixelSpacing': [2, 2, 2],
    'interpolator': 'sitkBSpline',
    'normalize': True,
    'normalizeScale': 100,
}
extractor = featureextractor.RadiomicsFeatureExtractor(**params)

# 断点续跑
done = set()
if out_csv.exists():
    old = pd.read_csv(out_csv)
    done = set(old['patient_id'].tolist())
    all_rows = old.to_dict('records')
    print(f"已有 {len(done)} 例完成，继续跑剩余的")
else:
    all_rows = []

img_files = sorted(img_dir.glob('*.nii.gz'))
total = len(img_files)
print(f"共 {total} 例，待处理 {total - len(done)} 例")

t0_all = time.time()
errors = []

for idx, img_path in enumerate(img_files):
    name = img_path.name.replace('.nii.gz', '')
    if name in done:
        continue
    mask_path = mask_dir / f'{name}.nii.gz'
    if not mask_path.exists():
        errors.append({'patient_id': name, 'error': 'mask_not_found'})
        print(f"[{idx+1}/{total}] {name} mask 不存在")
        continue

    try:
        t0 = time.time()
        img = sitk.Cast(sitk.ReadImage(str(img_path)), sitk.sitkFloat32)
        mask = sitk.Cast(sitk.ReadImage(str(mask_path)), sitk.sitkUInt8)
        result = extractor.execute(img, mask)
        row = {'patient_id': name}
        for k, v in result.items():
            if k.startswith('original_'):
                row[k] = float(v)
        all_rows.append(row)
        dt = time.time() - t0
        print(f"[{idx+1}/{total}] {name} 完成 {dt:.1f}s")

        # 每 20 例保存一次
        if len(all_rows) % 20 == 0:
            pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')

    except Exception as e:
        errors.append({'patient_id': name, 'error': str(e)[:100]})
        print(f"[{idx+1}/{total}] {name} 出错: {str(e)[:80]}")

# 最终保存
pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
if errors:
    pd.DataFrame(errors).to_csv(err_log, index=False, encoding='utf-8-sig')

elapsed = (time.time() - t0_all) / 60
print(f"\n全部完成，共 {len(all_rows)} 例，总用时 {elapsed:.1f} 分钟")
print(f"输出: {out_csv}")
if errors:
    print(f"错误日志: {err_log}")
