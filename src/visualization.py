from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import resolve_path, sanitize_name

def set_plot_style(config: dict) -> None:
    import matplotlib
    import matplotlib as mpl
    import seaborn as sns
    matplotlib.use('Agg', force=True)
    plot_cfg = config.get('plotting_settings', {})
    sns.set_theme(style=plot_cfg.get('style', 'whitegrid'), palette=plot_cfg.get('palette', 'colorblind'))
    mpl.rcParams.update({'font.family': plot_cfg.get('font_family', 'Arial'), 'font.size': plot_cfg.get('font_size', 9), 'axes.labelsize': plot_cfg.get('label_size', 10), 'axes.titlesize': plot_cfg.get('title_size', 11), 'legend.fontsize': plot_cfg.get('legend_size', 8), 'lines.linewidth': plot_cfg.get('line_width', 1.2), 'axes.linewidth': plot_cfg.get('line_width', 1.2), 'savefig.dpi': plot_cfg.get('dpi', 600), 'figure.dpi': plot_cfg.get('dpi', 600)})

def save_figure(fig, path_stem, config: dict) -> None:
    from pathlib import Path
    path_stem = Path(path_stem)
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    formats = config.get('plotting_settings', {}).get('formats', ['png', 'pdf', 'svg'])
    dpi = int(config.get('plotting_settings', {}).get('dpi', 600))
    fig.tight_layout()
    for fmt in formats:
        fig.savefig(path_stem.parent / f'{path_stem.name}.{fmt}', dpi=dpi, bbox_inches='tight')
