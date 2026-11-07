import sys
from pathlib import Path
import numpy as np
from pypdf import PdfReader
sys.path.insert(0,str(Path('outputs').resolve()))
from mahler_v3_figures import update_plot
d=np.load(Path('inputs')/'dataset_derived.npz')
predictor=dict(zip(d['k'],d['p']))
curves=[]
for run in (12,13,14):
 a=np.load(f'work/1000261_{run}.valid.npz')
 target=np.array([predictor[k] for k in a['k']])
 cols=-a['preds'][:,:301].T
 curves.append([np.corrcoef(v[m],target[m])[0,1] for v in cols for m in [np.isfinite(v)]])
curves=np.array(curves)
assert np.isfinite(curves).all() and curves.mean(0).argmax()==62
assert round(curves.mean(0)[300],3)==.876
update_plot(PdfReader(Path('inputs')/'figure_styles_25.pdf'),Path('outputs'),
 'v2_pearson_Phat_validation',35,24,.65,.05,[curves],[636],[1480],
 tick_indices=(648,682,894,928),min_path_points=250)
