"""Generate the corrected Section 5.1 plots from the archived prediction arrays."""
import sys
import json
from pathlib import Path
import numpy as np
from pypdf import PdfReader
from pypdf.generic import ContentStream, FloatObject

sys.path.insert(0, str(Path('outputs').resolve()))
from mahler_v3_figures import update_plot, write_form

out = Path('outputs')
pdf = PdfReader(Path('inputs')/'figure_styles_20.pdf')
groups = {}
for line in Path('work/metrics.jsonl').read_text().splitlines():
    name, _ = json.loads(line)
    groups.setdefault(name.split('/')[-2], []).append(name.split('/')[-1])

def validation(group):
    curves = []
    for run in groups[group]:
        a = np.load(f'work/{run}.valid.npz')
        ratio = a['preds'][:,:300] / (a['y'][:,0]/a['y'][:,1])[:,None]
        ok = (ratio > .5) & (ratio < 2)
        ok &= ~np.isclose(ratio, .5, atol=1e-14, rtol=0) & ~np.isclose(ratio, 2, atol=1e-14, rtol=0)
        curves.append(100*ok.mean(0))
    return np.asarray(curves)

for filename, page, form, groups_to_plot, bands, lines, step in [
    ('rk_root_number_validation', 22, 5, ('rk','rk_w'), [837,1451], [1758,2064], 10),
    ('rk_conductor_factorization_validation', 23, 7, ('rk_wN','rk_wfD','rk_wfN'), [824,1438,2052], [2252,2540,2829], 20),
]:
    update_plot(pdf, out, filename, page, form, 0, step,
                [validation(g) for g in groups_to_plot], bands, lines,
                tick_indices=(28,41,127,140), min_path_points=150)

# Use the same high-precision Mahler measure routine as the histogram audit.
exec(Path('work/hist.py').read_text().split('with zipfile.ZipFile')[0])
with zipfile.ZipFile(d/'mahler_uploads.zip') as z:
    name = next(n for n in z.namelist() if '/rk_wN/' in n and '/eval.test1.' in n)
    rows = list(json.loads(bz2.decompress(z.read(name))).values())
keys = {x['problem']['k']:4*math.pi**2*mu(x['problem']['k'])*x['problem']['w']/x['problem']['N'] for x in rows}
reference = [keys[x['problem']['k']]*x['answer'][1]/x['answer'][0] for x in rows]
predictions = []
for run, epoch in zip((18,19,20),(273,125,285)):
    a = np.load(f'work/1000262_{run}.test1.npz')
    values = np.array([keys[k] for k in a['k']])/a['preds'][:,epoch]
    assert len(values)==3000 and np.isfinite(values).all()
    predictions.append(values)
form = pdf.pages[23]['/Resources']['/XObject']['/Im9']
content = ContentStream(form,pdf)
ops = content.operations
# The template has 450 rectangles per series, with bins 0.00--9.00.
starts = [i for i,(a,op) in enumerate(ops) if op==b'W']
assert len(starts)==900
bins = np.arange(451)/50
counts = [np.histogram(reference,bins)[0], np.mean([np.histogram(v,bins)[0] for v in predictions],axis=0)]
base = float(ops[156][0][1])
scale = (float(ops[169][0][1])-base)/50
for start, count in zip(starts,np.concatenate(counts)):
    i = start+8
    assert [op for _,op in ops[i:i+6]] == [b'm',b'l',b'l',b'l',b'h',b'B']
    for j in (i+2,i+3):
        xy,op = ops[j]
        ops[j] = ([xy[0],FloatObject(base+scale*count)],op)
content.operations = ops
write_form(form,content,out/'rk_l2_distribution_test_corrected.pdf')
