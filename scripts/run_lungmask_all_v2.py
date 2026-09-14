import pandas as pd
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from lungmask import mask
import time
import csv

mf = pd.read_csv('/root/autodl-tmp/manifest.csv')

dataset_map = {
    "LUNG1": "01 LUNG1",
    "Radiogenomics": "02 Radiogenomics",
    "LungCT-Diagnosis": "03 LungCT-Diagnosis",
    "QIN-LUNG-CT": "04 QIN-LUNG-CT",
    "LUAD-CT-Survival": "05 LUAD-CT-Survival",
}

out_root = Path("/root/autodl-tmp/preprocessed")
(out_root / "images").mkdir(parents=True, exist_ok=True)
(out_root / "lung_masks").mkdir(parents=True, exist_ok=True)

log_path = out_root / "lungmask_log.csv"
log_rows = []
total = len(mf)
print(f"共 {total} 例，开始处理...")

def find_best_series(patient_dir):
    dcm_files = list(patient_dir.rglob('*.dcm'))
    if not dcm_files:
        return None
    series_dirs = {}
    for f in dcm_files:
        series_dirs.setdefault(f.parent, []).append(f)
    best = None
    best_size = 0
    for series_dir, files in series_dirs.items():
        try:
            reader = sitk.ImageSeriesReader()
            names = reader.GetGDCMSeriesFileNames(str(series_dir))
            if not names:
                continue
            reader.SetFileNames(names)
            img = reader.Execute()
            size = img.GetSize()[2]
            if size > best_size:
                best_size = size
                best = (series_dir, img)
        except Exception:
            continue
    return best

for i, row in mf.iterrows():
    dataset = row['dataset']
    pid = row['patient_id']
    cloud_folder = dataset_map[dataset]
    patient_dir = Path("/root/autodl-tmp") / cloud_folder / pid

    out_img = out_root / "images" / f"{dataset}_{pid}.nii.gz"
    out_mask = out_root / "lung_masks" / f"{dataset}_{pid}.nii.gz"

    if out_img.exists() and out_mask.exists():
        print(f"[{i+1}/{total}] {dataset}/{pid} 已存在，跳过")
        continue

    if not patient_dir.exists():
        log_rows.append({'dataset': dataset, 'patient_id': pid, 'status': 'no_dir', 'time_s': 0})
        print(f"[{i+1}/{total}] {dataset}/{pid} 目录不存在")
        continue

    t0 = time.time()
    try:
        result = find_best_series(patient_dir)
        if result is None:
            log_rows.append({'dataset': dataset, 'patient_id': pid, 'status': 'no_valid_series', 'time_s': 0})
            print(f"[{i+1}/{total}] {dataset}/{pid} 无可用 Series")
            continue

        series_dir, image = result
        segmentation = mask.apply(image)

        sitk.WriteImage(image, str(out_img))
        seg_img = sitk.GetImageFromArray(segmentation.astype(np.uint8))
        seg_img.CopyInformation(image)
        sitk.WriteImage(seg_img, str(out_mask))

        dt = time.time() - t0
        log_rows.append({'dataset': dataset, 'patient_id': pid, 'status': 'ok', 'time_s': round(dt, 1)})
        print(f"[{i+1}/{total}] {dataset}/{pid} 完成 {dt:.1f}s")

    except BaseException as e:
        log_rows.append({'dataset': dataset, 'patient_id': pid, 'status': f'error: {str(e)[:80]}', 'time_s': 0})
        print(f"[{i+1}/{total}] {dataset}/{pid} 出错: {str(e)[:80]}")

with open(log_path, 'w', newline='', encoding='utf-8-sig') as fp:
    writer = csv.DictWriter(fp, fieldnames=['dataset', 'patient_id', 'status', 'time_s'])
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\n全部完成，日志保存到 {log_path}")
df_log = pd.DataFrame(log_rows)
print("\n状态统计:")
print(df_log['status'].value_counts())
