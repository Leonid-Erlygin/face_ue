"""Create separate diagnostic figures from the saved tables; no plotting-only refits."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.mprisk_evidence.study import slug


def make_plots(root):
    root=Path(root);out=root/'plots';out.mkdir(exist_ok=True)
    file=root/'tables'/'rejection_curves.csv'
    if not file.is_file():return
    df=pd.read_csv(file)
    methods=['MPRisk','MPRisk calibrated','MPRisk-paper retuned','HolUE (refit fixed decisions)',
             'Linear all baseline scores','Point-vMF (NLL fitted)']
    df=df[(df['split']=='test') & df['method'].isin(methods)]
    for (dataset,fpir),group in df.groupby(['dataset','target_fpir']):
        for metric,label in [('f1','F1 (repository definition)'),('risk','Observed error rate'),('fpir','FPIR'),('fnir','FNIR')]:
            fig,ax=plt.subplots(figsize=(7.2,4.8))
            for method in methods:
                g=group[group.method==method].sort_values('filter_fraction')
                if not g.empty:ax.plot(g.filter_fraction,g[metric],label=method)
            ax.set_xlabel('Fraction filtered');ax.set_ylabel(label)
            ax.set_title(f'{dataset}; target FPIR = {fpir:g}; fixed decisions')
            ax.legend(fontsize=7);ax.grid(alpha=.3);fig.tight_layout()
            fig.savefig(out/f'{slug(dataset)}_fpir_{fpir:g}_{metric}.png',dpi=160);plt.close(fig)
    path=root/'tables'/'reliability_bins.csv'
    if path.is_file():
        df=pd.read_csv(path);df=df[(df['split']=='test')&(df['method'].isin(['MPRisk','MPRisk calibrated']))]
        for (dataset,fpir),group in df.groupby(['dataset','target_fpir']):
            fig,ax=plt.subplots(figsize=(5.5,5))
            ax.plot([0,1],[0,1],linestyle='--',label='Perfect calibration')
            for method in ['MPRisk','MPRisk calibrated']:
                g=group[(group.method==method)&(group['count']>0)].sort_values('predicted')
                ax.plot(g.predicted,g.observed,marker='o',label=method)
            ax.set(xlabel='Predicted error probability',ylabel='Observed error frequency',title=f'{dataset}; FPIR {fpir:g}')
            ax.legend(fontsize=8);ax.grid(alpha=.3);fig.tight_layout()
            fig.savefig(out/f'{slug(dataset)}_fpir_{fpir:g}_reliability.png',dpi=160);plt.close(fig)
