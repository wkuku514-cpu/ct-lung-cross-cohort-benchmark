import torch
import torchvision.models as models
import torchvision.transforms as transforms
import SimpleITK as sitk
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# 加载 InceptionV3
print("加载 InceptionV3...")
model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
model.fc = torch.nn.Identity()
model.eval()
model = model.to(device)

preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_embedding(img_array, mask_array, n_slices=5):
    """从 CT + mask 提取 2048 维嵌入"""
    z_layers = np.where(mask_array.sum(axis=(1,2)) > 0)[0]
    if len(z_layers) == 0:
        return None
    if len(z_layers) >= n_slices:
        indices = np.linspace(0, len(z_layers)-1, n_slices).astype(int)
        selected = z_layers[indices]
    else:
        selected = z_layers

    embeddings = []
    for z in selected:
        slice_hu = img_array[z]
        windowed = np.clip(slice_hu, -1000, 400)
        windowed = ((windowed + 1000) / 1400 * 255).astype(np.uint8)
        img_rgb = np.stack([windowed]*3, axis=-1)
        img_pil = Image.fromarray(img_rgb)
        img_t = preprocess(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(img_t).cpu().numpy().flatten()
        embeddings.append(feat)
    return np.mean(embeddings, axis=0)

# 路径
img_dir = Path('/root/autodl-tmp/preprocessed/images')
mask_dir = Path('/root/autodl-tmp/preprocessed/lung_masks')
out_dir = Path('/root/autodl-tmp/preprocessed/features')
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / 'deep_embeddings.csv'

# 断点续跑
done = set()
if out_csv.exists():
    old = pd.read_csv(out_csv)
    done = set(old['patient_id'].tolist())
    all_rows = old.to_dict('records')
    print(f"已有 {len(done)} 例完成")
else:
    all_rows = []

img_files = sorted(img_dir.glob('*.nii.gz'))
total = len(img_files)
print(f"共 {total} 例")

t0_all = time.time()
for idx, img_path in enumerate(img_files):
    name = img_path.name.replace('.nii.gz', '')
    if name in done:
        continue
    mask_path = mask_dir / f'{name}.nii.gz'
    if not mask_path.exists():
        print(f"[{idx+1}/{total}] {name} mask 不存在")
        continue

    try:
        t0 = time.time()
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(img_path)))
        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))
        emb = extract_embedding(img, mask, n_slices=5)
        if emb is None:
            print(f"[{idx+1}/{total}] {name} 无有效切片")
            continue
        row = {'patient_id': name}
        for i, v in enumerate(emb):
            row[f'emb_{i:04d}'] = float(v)
        all_rows.append(row)
        dt = time.time() - t0
        if (idx+1) % 20 == 0 or idx < 3:
            print(f"[{idx+1}/{total}] {name} 完成 {dt:.2f}s")

        if len(all_rows) % 50 == 0:
            pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
    except Exception as e:
        print(f"[{idx+1}/{total}] {name} 出错: {str(e)[:80]}")

pd.DataFrame(all_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
print(f"\n完成，共 {len(all_rows)} 例，总用时 {(time.time()-t0_all)/60:.1f} 分钟")
print(f"输出: {out_csv}")
