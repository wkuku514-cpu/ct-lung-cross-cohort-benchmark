# T8 figure rework via the scipilot-figure-skill workflow:
# profile -> select -> spec -> style -> plot -> self-check loop -> export.
# Style targets (user 2026-09-14): Arial/Helvetica; axis label 8 pt; ticks 7 pt;
# panel labels 10 pt bold lowercase; Okabe-Ito; >=300 dpi; PNG + vector PDF.
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from PIL import Image

SKILL = r'C:\Users\kuku\Desktop\cancer-imaging-ct\.agents\skills\scipilot-figure-skill\scripts'
sys.path.insert(0, SKILL)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from setup_style import setup_style
from layout_tools import finalize_figure, add_panel_labels
from visual_qa import render_preview, audit_layout, print_report
import make_figures_v2 as core

ROOT, RUN, FIG = core.ROOT, core.RUN, core.FIG
PREV = os.path.join(FIG, '_preview')
os.makedirs(PREV, exist_ok=True)
SRC = os.path.join(FIG, 'source_data')
os.makedirs(SRC, exist_ok=True)
TIMES, THR = core.TIMES, core.THR

BLUE, ORANGE, GREEN, GREY = '#0072B2', '#D55E00', '#009E73', '#999999'
BOX_FILL, BOX_EDGE = '#F5F5F5', '#333333'


def save_src(df, name):
    """Persist the numbers behind a figure (provenance requirement)."""
    p = os.path.join(SRC, name + '.csv')
    df.to_csv(p, index=False, encoding='utf-8-sig')
    print('source data ->', p)


def style():
    setup_style(journal='nature', lang='en')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'axes.linewidth': 0.75, 'xtick.major.width': 0.75, 'ytick.major.width': 0.75,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'lines.linewidth': 0.75,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
        'axes.unicode_minus': False,
    })


def finish(fig, name, n_panels=None, panel_fs=10, panel_axes=None):
    finalize_figure(fig)
    if n_panels:
        add_panel_labels(fig, axes=panel_axes, style='nature', fontsize=panel_fs,
                         fontweight='bold')
    render_preview(fig, os.path.join(PREV, name + '_preview.png'), dpi=150)
    issues = audit_layout(fig)
    print_report(issues)
    for ext, kw in (('.png', dict(dpi=320)), ('.pdf', {})):
        p = os.path.join(FIG, name + ext)
        try:
            # bbox_inches=None keeps the exact figsize so that point sizes are the
            # true final-size sizes (scipilot hard rule 1: never rescale).
            fig.savefig(p, bbox_inches=None, **kw)
        except PermissionError:
            print('WARNING locked, kept existing ->', p)
    try:
        Image.open(os.path.join(FIG, name + '.png')).convert('L').save(
            os.path.join(PREV, name + '_grayscale.png'))
    except Exception as exc:
        print('grayscale skipped:', exc)
    plt.close(fig)
    nfail = len([i for i in issues if i[0] == 'FAIL'])
    print('=== %s exported (FAIL=%d) ===' % (name, nfail))
    return issues


def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

    def box(x, y, w, h, text, fs=6.6, blue=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=1.0',
                                    fc=BOX_FILL, ec=BOX_EDGE, lw=0.75, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs,
                color=BLUE if blue else '#000000',
                fontweight='bold' if blue else 'normal', zorder=3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=7, color='#333333', lw=0.75, zorder=1))

    cols = ['NSCLC-Radiomics\n(LUNG1)\nn = 422', 'NSCLC-Radiogenomics\nn = 211',
            'LungCT-Diagnosis\nn = 61', 'QIN-LUNG-CT\nn = 47', 'LUAD-CT-Survival\nn = 40']
    x0, w, gap = 2.0, 18.4, 1.5
    for i, t in enumerate(cols):
        box(x0 + i * (w + gap), 80, w, 14, t)
    # Merge all five data sources before reporting the imaging-entry total.
    centers = [x0 + i * (w + gap) + w / 2 for i in range(len(cols))]
    branch_y = 75.5
    for xc in centers:
        ax.plot([xc, xc], [79, branch_y], color='#333333', lw=0.75, zorder=1)
    ax.plot([centers[0], centers[-1]], [branch_y, branch_y],
            color='#333333', lw=0.75, zorder=1)
    arrow(50, branch_y, 50, 72)
    box(26, 60, 48, 11, 'Imaging entries\nn = 781', blue=True)
    arrow(50, 59, 50, 52)
    box(26, 40, 48, 11, 'Distinct original patient IDs\nn = 741', blue=True)
    arrow(50, 39, 50, 32)
    box(20, 20, 60, 11, 'Survival analysis cohort\nn = 633', blue=True)
    ax.text(50, 12, 'Sensitivity: counting 24 additional candidate UID-suffix links as shared gives n = 717.',
            ha='center', fontsize=6.2, color='#333333')
    ax.text(50, 5, 'LUNG1 n = 422 (development)   |   Radiogenomics n = 211 (target evaluation)\n'
            'LungCT-Diagnosis (61), QIN-LUNG-CT (47), and LUAD-CT-Survival (40): not used to fit the OS model.',
            ha='center', fontsize=5.8, color='#333333')
    return finish(fig, 'Fig1', n_panels=1)


def fig2():
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    mf = pd.read_csv(os.path.join(RUN, 'master_features_clean.csv'),
                     encoding='utf-8-sig', low_memory=False)
    cb = pd.read_csv(os.path.join(RUN, 'surv_combat.csv'), encoding='utf-8-sig')

    def pca2(df, cols):
        X = StandardScaler().fit_transform(
            SimpleImputer(strategy='median').fit_transform(df[cols].values))
        return PCA(n_components=2, random_state=0).fit_transform(X)

    sm = mf[mf['dataset'].isin(['LUNG1', 'Radiogenomics'])].dropna(
        subset=['survival_time_days', 'event'])
    P = {
        'a': (pca2(sm, [c for c in mf.columns if c.startswith('original_')]),
              (sm['dataset'] == 'LUNG1').values),
        'b': (pca2(cb, [c for c in cb.columns if c.startswith('rad_combat_')]),
              (cb['dataset'] == 'LUNG1').values),
        'c': (pca2(sm, [c for c in mf.columns if c.startswith('emb_')]),
              (sm['dataset'] == 'LUNG1').values),
        'd': (pca2(cb, [c for c in cb.columns if c.startswith('emb_combat_')]),
              (cb['dataset'] == 'LUNG1').values),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4), constrained_layout=True)
    keys = [['a', 'b'], ['c', 'd']]
    import pandas as _pd
    recs = []
    for r in range(2):
        xs = np.concatenate([P[k][0][:, 0] for k in keys[r]])
        ys = np.concatenate([P[k][0][:, 1] for k in keys[r]])
        xp, yp = 0.05 * (xs.max() - xs.min()), 0.05 * (ys.max() - ys.min())
        for c, k in enumerate(keys[r]):
            ax = axes[r][c]
            xy, is_l1 = P[k]
            for (x, y), flag in zip(xy, is_l1):
                recs.append({'panel': k, 'block': 'radiomics' if r == 0 else 'embeddings',
                             'cohort': 'LUNG1' if flag else 'Radiogenomics',
                             'PC1': float(x), 'PC2': float(y)})
            ax.scatter(xy[is_l1, 0], xy[is_l1, 1], s=12, alpha=0.6, color=BLUE,
                       marker='o', edgecolor='none')
            ax.scatter(xy[~is_l1, 0], xy[~is_l1, 1], s=12, alpha=0.6, color=ORANGE,
                       marker='^', edgecolor='none')
            ax.set_xlim(xs.min() - xp, xs.max() + xp)
            ax.set_ylim(ys.min() - yp, ys.max() + yp)
            ax.set_xlabel('PC1')
            if c == 0:
                ax.set_ylabel('PC2 (%s)' % ('radiomics' if r == 0 else 'embeddings'))
            if r == 0:
                ax.set_title(['Before ComBat', 'After ComBat'][c], fontsize=8)
    handles = [Line2D([], [], marker='o', ls='none', color=BLUE, markersize=4,
                      label='LUNG1 (n = 422)'),
               Line2D([], [], marker='^', ls='none', color=ORANGE, markersize=4,
                      label='Radiogenomics (n = 211)')]
    fig.legend(handles=handles, loc='outside lower center', ncol=2, frameon=False)
    save_src(_pd.DataFrame(recs), 'fig2_pca_coordinates')
    return finish(fig, 'Fig2', n_panels=4)


def _grid33(figsize):
    """3x3 data panels plus a dedicated narrow left column for row labels."""
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    gs = fig.add_gridspec(3, 4, width_ratios=[0.05, 1, 1, 1], wspace=0.06, hspace=0.08)
    axes = np.empty((3, 3), dtype=object)
    lab_axes = []
    for r in range(3):
        la = fig.add_subplot(gs[r, 0]); la.axis('off'); lab_axes.append(la)
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c + 1])
    return fig, axes, lab_axes


def fig3(data):
    fig, axes, lab_axes = _grid33((7.2, 7.2))
    import pandas as _pd
    recs = []
    rows = [('External\noriginal', data['cal_orig'], BLUE, False),
            ('External\nrecalibrated', data['cal_recal'], ORANGE, True),
            ('Internal LUNG1\nout-of-fold', data['cal_int'], BLUE, False)]
    for r, (label, dset, col, overlay) in enumerate(rows):
        lab_axes[r].text(0.5, 0.5, label, rotation=90, ha='center', va='center', fontsize=7)
        for c, tt in enumerate(TIMES):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], ls='--', lw=0.75, color='#000000', zorder=1)
            p, o = dset[tt]
            for gi, (pv, ov) in enumerate(zip(p, o)):
                recs.append({'condition': label.replace('\n', ' '), 'day': tt,
                             'risk_quartile': gi + 1, 'predicted_survival': float(pv),
                             'observed_survival_KM': float(ov)})
            ax.scatter(p, o, s=18, color=col, edgecolor='none', zorder=3)
            if overlay:
                p0, o0 = data['cal_orig'][tt]
                ax.scatter(p0, o0, s=16, facecolors='none', edgecolors=BLUE,
                           linewidths=0.75, zorder=2)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
            if r == 0:
                ax.set_title('%d days' % tt, fontsize=8)
            if r == 2:
                ax.set_xlabel('Predicted survival')
    handles = [Line2D([], [], marker='o', ls='none', color=BLUE, markersize=4,
                      label='Original, external'),
               Line2D([], [], marker='o', ls='none', color=ORANGE, markersize=4,
                      label='Recalibrated, external'),
               Line2D([], [], marker='o', ls='none', markerfacecolor='none',
                      markeredgecolor=BLUE, markersize=4, label='Original overlaid'),
               Line2D([], [], ls='--', color='#000000', lw=0.75, label='Ideal')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
    save_src(_pd.DataFrame(recs), 'fig3_calibration_points')
    return finish(fig, 'Fig3', n_panels=9, panel_axes=list(axes.ravel()))


def fig4(data):
    fig, axes, lab_axes = _grid33((7.2, 7.2))
    import pandas as _pd
    recs = []
    d = data['dca']; thr = np.array(data['thr'])
    for r in range(3):
        lab_axes[r].text(0.5, 0.5, ['External\noriginal', 'External\nrecalibrated',
                                    'Internal LUNG1'][r], rotation=90, ha='center',
                         va='center', fontsize=7)
        for c, tt in enumerate(TIMES):
            ax = axes[r][c]
            if r == 0:
                ax.plot(thr, d['ext_orig_deep'][tt], color=BLUE, lw=1.25)
                ax.plot(thr, d['ext_orig_joint'][tt], color=ORANGE, lw=1.25)
            elif r == 1:
                ax.plot(thr, d['ext_recal_joint'][tt], color=ORANGE, lw=1.25)
            else:
                ax.plot(thr, d['int_joint'][tt], color=ORANGE, lw=1.25)
            series = []
            if r == 0:
                series = [('Deep', 'ext_orig_deep'), ('Joint', 'ext_orig_joint')]
            elif r == 1:
                series = [('Joint', 'ext_recal_joint')]
            else:
                series = [('Joint', 'int_joint')]
            for model, key in series:
                for th, nb in zip(thr, d[key][tt]):
                    recs.append({'condition': ['External original', 'External recalibrated',
                                               'Internal LUNG1'][r], 'model': model, 'day': tt,
                                 'threshold': float(th), 'net_benefit': float(nb)})
            ax.axhline(0, color=GREY, lw=0.75, zorder=1)
            ax.set_xlim(0, 0.9); ax.set_ylim(-0.2, 0.5)
            ax.set_xticks([0, 0.3, 0.6, 0.9]); ax.set_yticks([-0.2, 0, 0.2, 0.4])
            if r == 0:
                ax.set_title('%d days' % tt, fontsize=8)
            if r == 2:
                ax.set_xlabel('Threshold probability')
    handles = [Line2D([], [], color=BLUE, lw=1.25, label='Deep'),
               Line2D([], [], color=ORANGE, lw=1.25, label='Joint'),
               Line2D([], [], color='#000000', ls='--', lw=0.75, label='Treat all'),
               Line2D([], [], color=GREY, lw=0.75, label='Treat none')]
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False)
    save_src(_pd.DataFrame(recs), 'fig4_dca_curves')
    return finish(fig, 'Fig4', n_panels=9, panel_axes=list(axes.ravel()))


def fig5():
    import pandas as pd
    cs = pd.read_csv(os.path.join(ROOT, 'pack', 'coef_stability.csv'))
    save_src(cs[['component', 'coef_LUNG1', 'coef_Radiogenomics', 'cv_stability']],
             'fig5_coefficients')
    top = set(cs.reindex(cs['coef_LUNG1'].abs().sort_values(ascending=False).index)
              .head(10)['component']) | \
          set(cs.reindex(cs['coef_Radiogenomics'].abs().sort_values(ascending=False).index)
              .head(10)['component'])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.0, 1.15]})
    # (a) all 89 components, with the top-10 union highlighted
    ax = axes[0]
    is_top = cs['component'].isin(top)
    # shape is the redundant cue so the two classes stay separable in grayscale
    ax.scatter(cs.loc[~is_top, 'coef_LUNG1'], cs.loc[~is_top, 'coef_Radiogenomics'],
               s=12, color=GREY, marker='o', edgecolor='none', label='Other components')
    ax.scatter(cs.loc[is_top, 'coef_LUNG1'], cs.loc[is_top, 'coef_Radiogenomics'],
               s=28, color=ORANGE, marker='s', edgecolor='none', label='Top-10 components')
    lim = max(cs['coef_LUNG1'].abs().max(), cs['coef_Radiogenomics'].abs().max()) * 1.15
    ax.plot([-lim, lim], [-lim, lim], ls='--', lw=0.75, color='#000000', label='Identity')
    ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Cox coefficient, LUNG1')
    ax.set_ylabel('Cox coefficient, Radiogenomics')
    ax.text(0.97, 0.97, 'Spearman \u03c1 = 0.223 (descriptive)\nTop-10 overlap: 10% (1/10)',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.6,
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=GREY, lw=0.5))
    ax.legend(loc='lower right', frameon=False, fontsize=6.5)
    # (b) the ten largest LUNG1 coefficients with their Radiogenomics counterparts
    ax2 = axes[1]
    sub = cs.reindex(cs['coef_LUNG1'].abs().sort_values(ascending=False).index).head(10)
    ypos = np.arange(len(sub))[::-1]
    h = 0.38
    ax2.barh(ypos + h / 2, sub['coef_LUNG1'], height=h, color=BLUE,
             edgecolor='#333333', linewidth=0.3, label='LUNG1')
    ax2.barh(ypos - h / 2, sub['coef_Radiogenomics'], height=h, color=GREEN,
             edgecolor='#333333', linewidth=0.3, hatch='///', label='Radiogenomics')
    ax2.axvline(0, color='#000000', lw=0.6)
    ax2.set_yticks(ypos)
    ax2.set_yticklabels(sub['component'], fontsize=6.5)
    ax2.set_xlabel('Cox coefficient')
    ax2.legend(loc='lower right', frameon=False, fontsize=6.5)
    return finish(fig, 'Fig5', n_panels=2, panel_axes=list(axes))


if __name__ == '__main__':
    style()
    issues_all = {}
    issues_all['Fig1'] = fig1()
    issues_all['Fig2'] = fig2()
    data = core.compute_all()
    issues_all['Fig3'] = fig3(data)
    issues_all['Fig4'] = fig4(data)
    issues_all['Fig5'] = fig5()
    with open(os.path.join(PREV, 'audit_program.json'), 'w') as fh:
        json.dump(issues_all, fh, default=str, indent=1)
    print('v3 done')
