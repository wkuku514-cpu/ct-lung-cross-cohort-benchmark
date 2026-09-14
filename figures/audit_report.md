# T8 图件技术审核报告

审核日期：2026-09-14　｜　工具：scipilot-figure-skill v2.1（程序自检 + AI 读图）+ read-image（视觉大模型）
审核范围：`figures/Fig1.png–Fig5.png`、`figures/Fig1.pdf–Fig5.pdf`

## 1. 本会话视觉能力说明

**本会话不支持原生图像输入**（`view_image` 被系统拒绝）。因此"AI 读图"环节通过 `read-image` skill 实现：把 PNG 交给视觉大模型（阿里云百炼 Qwen-VL）取回描述，再对照 `references/visual_review.md` 的 8 项清单逐条核对。**所有"读图结论"均来自该外呼模型，不是我的原生视觉判断**，证据为每张图的调用记录。

## 2. 技术规格（实测）

| 图 | PNG 像素 | PNG 体积 | 有效分辨率* | PNG 元数据 DPI | PDF 尺寸 | PDF 字体（已嵌入） |
|---|---|---|---|---|---|---|
| Fig1 | 2164 × 1624 | 119 KB | 301 dpi | 300 | 7.22 × 5.42 in | Arial, Arial-BoldMT |
| Fig2 | 2164 × 1922 | 506 KB | 301 dpi | 300 | 7.22 × 6.41 in | Arial, Arial-BoldMT |
| Fig3 | 2164 × 2162 | 248 KB | 301 dpi | 300 | 7.22 × 7.21 in | Arial, Arial-BoldMT |
| Fig4 | 2164 × 2162 | 237 KB | 301 dpi | 300 | 7.22 × 7.21 in | Arial, Arial-BoldMT |
| Fig5 | 2164 × 1082 | 167 KB | 301 dpi | 300 | 7.22 × 3.61 in | Arial, Arial-BoldMT |

\* 有效分辨率 = PNG 宽度像素 ÷ 7.2 in（设定的最终宽度）。

**结论**：分辨率 ≥300 dpi ✅；同时提供矢量 PDF ✅；字体为 Arial 且子集嵌入 ✅；PDF 尺寸精确等于设定宽度（未被 tight bbox 缩放）✅。

## 3. 顶刊硬约束逐项核对

| 约束 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 字体 | Arial 或 Helvetica | Arial（rcParams + PDF 字体表确认） | ✅ |
| 轴标签 | 8 pt | `axes.labelsize = 8` | ✅ |
| 刻度 | 7 pt | `xtick/ytick.labelsize = 7` | ✅ |
| 面板标签 | 10 pt bold 小写 a,b,c | `add_panel_labels(style='nature', fontsize=10, fontweight='bold')`，AI 读图确认对齐 | ✅ |
| 面板标签风格一致 | 全文一种风格 | 全部小写加粗（无 (a)/A 混用） | ✅ |
| 配色 | Okabe-Ito，禁红绿 | 蓝 #0072B2 / 橙 #D55E00 / 灰 #999999，无红绿对比 | ✅ |
| 色图 | 禁 rainbow/jet | 未使用任何色图（无热力图） | N/A |
| 分辨率 | ≥300 dpi | 301 dpi（有效） | ✅ |
| 矢量输出 | PDF | 5 张 PDF 齐备 | ✅ |
| 轴不夸大差异 | 允许比例尺 | 标定/决策曲线固定 0–1 与 −0.2–0.5；PCA 为共享主成分轴（见下注） | ✅ |
| 说明 n | 图内或图注标明 | 图内标注 n（Fig2 图例、Fig1 方框），图注全部写明 n | ✅ |

**PCA 轴注**：Fig2 的 a/b 与 c/d 分别共享同一坐标范围，但**不对跨行（影像组学 vs 嵌入）共用坐标轴**——两组特征的主成分尺度不可比，强行统一会误导。此处按"同一量在子图间同尺度"的原则处理。

## 4. 自检闭环记录（3 轮）

| 轮次 | 触发 | 发现 | 处置 |
|---|---|---|---|
| R1 | 程序自检 | Fig3/Fig4 报告行标签可能越界 | 先核验导出成品（当时用 tight bbox），AI 读图判定"完整可见"→ 定性为预览假阳性 |
| R2 | 技术规格核查 | **PDF 被 tight bbox 裁小**（Fig1 仅 5.78 in ≠ 设定 7.2 in），违反"按最终尺寸出图"硬规则 | 图例改用 `loc='outside lower center'`（constrained_layout 预留空间），导出改 `bbox_inches=None` |
| R3 | 程序自检 + AI 读图 | Fig3 行标签 "OOF" 被左缘轻微切掉（程序与 AI 一致） | 改为 GridSpec 专用左侧标签列，行标签置于独立子图；面板字母改为显式传入 9 个数据子图，避免标签列被误编号 |

第 3 轮后：**程序自检 5/5 PASS（无 FAIL、无 WARN）**，AI 读图 5/5 通过。

## 5. AI 读图复核结果（对照 visual_review.md 8 项）

| 检查项 | Fig1 | Fig2 | Fig3 | Fig4 | Fig5 |
|---|---|---|---|---|---|
| 1 缺字/方框 | 无 | 无 | 无 | 无 | 无 |
| 2 文字裁切 | 无 | 无 | 无 | 无 | 无 |
| 3 图例压数据 | 无图例，不适用 | 未压 | 未压 | 未压 | 未压 |
| 4 面板字母对齐 | a 正确 | a–d 齐全 | a–i 横竖对齐 | a–i 横竖对齐 | a、b 正确 |
| 5 子图相互侵入 | 不适用 | 无 | 无 | 无 | 无 |
| 6 配色可区分 | 蓝/灰/黑 | 蓝圆 vs 橙三角（双重编码） | 蓝/橙/黑虚线 | 蓝/橙/黑/灰 | 橙/灰/黑虚线 |
| 7 数据未被切 | 不适用 | 未切 | 未切 | 未切 | 未切 |
| 8 跨子图一致 | 不适用 | 行内同尺度同色 | 行内同尺度 | 行内同尺度 | 同一量同色 |

## 6. 已知限制与需人工确认

1. **AI 读图是外呼模型的描述，不等于人类目视终检**。建议投稿前由作者亲自看一遍 5 张 PNG，特别确认 Fig3/Fig4 中蓝橙两色的实际观感。
2. **Fig4 第 2、3 行只有 Joint 曲线**（无 Deep）——因原始分析只对联合模型做了 recalibration 与内部 OOF 决策曲线。图注已明确说明；若需补 Deep 的对应曲线，属于新增分析，需你批准。
3. **字号与 Nature 规范的差异**：Nature 建议标签 5–7 pt，你指定的是 8 pt。本稿按你的指定执行（双栏 7.2 in 下 8 pt 可读）。若最终投 Nature 系，需整体下调 1 pt 再复核。
4. **彩色 vs 灰度**：已生成灰度预览（`figures/_preview/*_grayscale.png`）供色盲/黑白检查，但我未对其做 AI 读图复核；如需，可补一轮。
5. **期刊未锁定**：本报告按"双栏 7.2 in + Arial + 8/7/10 pt"通用顶刊规格出具；定刊后若栏宽或字号要求不同，需重导（脚本已参数化）。

## 7. 图源数据留存（provenance）

按 `scientific-visualization` 的要求（"保留源数据与转换过程、提供输出溯源"），每张数据图的底层数值已单独导出，随图一起交付：

| 文件 | 内容 | 生成方式 |
|---|---|---|
| `source_data/fig2_pca_coordinates.csv` | 4 个面板全部 PCA 坐标（面板、特征块、队列、PC1、PC2） | 由 `master_features_clean.csv` / `surv_combat.csv` 重算 |
| `source_data/fig3_calibration_points.csv` | 3 条件 × 3 时间点 × 4 风险分组的预测值与 KM 观测值 | 由 LUNG1/外部队列重新拟合 Cox 后计算 |
| `source_data/fig4_dca_curves.csv` | 3 条件 × 时间点 × 阈值的净获益曲线 | 同上的模型重新计算 |
| `source_data/fig5_coefficients.csv` | 89 个分量的双队列 Cox 系数与 CV 稳定性 | 复制自 `coef_stability.csv` |

**重要说明**：Fig3/Fig4 的数值由我在本机用 `lifelines 0.30.3` 重新拟合得到（原始分析环境为 `lifelines 0.30.0`）。图注与 Results 中引用的数字仍以原始分析输出（T1 表）为准；本机重算仅用于生成图形几何，不用于替换任何报告数值。二者的一致性已核实：本机重算的临床模型点估计为 0.594 / 训练 0.542，与 T1 记录逐位一致。

## 8. 第二轮闭环：Fig5 冗余与灰度复核（2026-09-14 追加）

按用户裁决执行，共两轮修改：

| 轮次 | 触发 | 发现 | 处置 |
|---|---|---|---|
| R4 | 用户质疑 Fig5 的 a/b 是否真的不同 | **AI 读图确认两面板坐标范围与内容相同 → panel b 冗余** | 采纳"方案 B"：panel a 保留全量散点（89 分量，Top-10 联合集橙色方形、其余灰色圆形）；panel b 改为**LUNG1 系数最大的 10 个分量**的两队列分组水平条形图（LUNG1 蓝色实心、NSCLC-Radiogenomics 绿色斜线）。避免颜色语义冲突：panel a 的橙色专表"Top-10"，panel b 用蓝/绿专表队列 |
| R5 | 灰度/色盲 AI 复核 | **Fig5 panel a 的橙点与灰点在灰度下无法区分**（读图模型明确报告） | 增加**形状冗余编码**：其余分量为圆形、Top-10 为方形；panel b 的绿色柱加斜线纹理。复读确认两类点在灰度下可区分 |

**灰度复核结果（AI 读图 `figures/_preview/*_grayscale.png`）**

| 图 | 关键问题的可区分性 | 文字可读性 | 判定 |
|---|---|---|---|
| Fig1 | 方框与箭头清晰可辨 | 全部可读，含面板字母 a 与方框数字 | ✅ |
| Fig2 | 圆形（LUNG1）vs 三角形（NSCLC-Radiogenomics）灰度下可区分 | 全部可读 | ✅ |
| Fig3 | 实心点、空心点与虚线（ideal）可区分 | 全部可读，无裁切 | ✅ |
| Fig4 | 实线（Deep/Joint）、虚线（Treat-all）、细线（Treat-none）可区分 | 全部可读 | ✅ |
| Fig5 | 修正后可区分：圆形 vs 方形（panel a）、实心 vs 斜线（panel b）；修正前 panel a 不可区分 | 全部可读 | ✅（R5 修正后） |

**Fig5 新版复核结论**：AI 读图确认 panel a 为二维散点图、panel b 为一维条形图，**两面板图形明显不同**；panel b 每个分量名对应两根柱（蓝= LUNG1，绿+斜线= Radiogenomics）；无文字裁切，图例未压数据。

## 9. 与正文的同步更新

- T4 Methods 2.5 段末已按裁决加入 Code availability 句（GitHub + MIT + Zenodo DOI）。
- T8_declarations.md 的 Code availability 段已加入 Zenodo DOI `10.5281/zenodo.22752604`。
- 因加入该句，Methods 由 1,520 词增至约 1,559 词，超出此前 1,520 的放宽上限；该句为用户指定文本，未作删减，如需回到上限内需另行指示压缩位置。
