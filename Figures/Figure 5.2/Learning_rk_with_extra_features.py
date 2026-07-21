#!/usr/bin/env python
# coding: utf-8

# In[4]:


# plot_PB_compare.sage
#
# Compare the PB experiments of exp 3, 6, 67, which all learn r_k = [sign, |n_k|]
# from the SAME inputs k, w plus ONE extra feature:
#     exp 3   -> raw conductor  N        (two runs: PB3, PB6)
#     exp 6   -> factored conductor  N   (N_Factor)
#     exp 67  -> factored discriminant Delta  (D_Factor)
#
# "size accuracy" is the criterion for learning r_k: an output s_k is correct iff
# it has the same sign as r_k and  |r_k|/2 < |s_k| < 2|r_k|.  The eval stores the
# magnitude |n_k| = 1/|r_k|, and s <-> 1/s preserves the factor-2 window, so with
# ACCEPT = 2 the test  |log(guess) - log(true)| < log(ACCEPT)  is exactly this.
# Sign accuracy is 100% throughout (w is given), so only size accuracy is plotted.
#
# Run:  sage plot_PB_compare.sage
import os, re, glob, math
import numpy as np

# ===================== configure =====================
# label, folder (relative to this script, or absolute), colour, linestyle, linewidth
HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
RUNS = [
    (r'conductor $N_k$',        '../MahlerExperiments/rk/rk_wN/final-03e.PB3-exp',  '#555555', '-',  2.0),
#    (r'exp 3 -- raw conductor $N$ (run B)',        '../MahlerExperiments/rk/rk_wN/final-03e.PB6-exp',  '#999999', '--', 1.6),
    (r'factored conductor $\mathrm{factor}(N_k)$',           '../MahlerExperiments/rk/rk_wfN/size/rk_wfN (size)',  '#1f77b4', '-',  2.0),
    (r'factored discriminant $\mathrm{factor}(\Delta_k)$',  '../MahlerExperiments/rk/rk_wfD/size/rk_wfD (size)','#d62728','-',  2.0),
]
ACCEPT = 2.0          # magnitude tolerance factor
YMIN   = 30           # y-axis lower limit (%)
# =====================================================

def epoch_of(p):
    return int(p.rsplit('.', 1)[1])

def locate(folder):
    """Find the experiment folder: as given, under HERE, or anywhere below HERE."""
    for c in [folder, os.path.join(HERE, folder)]:
        if [f for f in glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()]:
            return c
    hits = [f for f in glob.glob(os.path.join(HERE, '**', os.path.basename(folder),
                                  'eval.valid.elliptic_curve.*'), recursive=True) if f.rsplit('.', 1)[-1].isdigit()]
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError(folder)

# answer=[a0,a1] ... (greedy) decoded=[g0,g1]
BLOCK = re.compile(
    r"answer=\[(?P<a0>-?\d+),\s*(?P<a1>-?\d+)\].*?greedy\s*\n\s*decoded=\[(?P<g0>-?\d+),\s*(?P<g1>-?\d+)\]",
    re.S)
logA = math.log(ACCEPT)

def curve(folder):
    files = sorted([f for f in glob.glob(os.path.join(locate(folder), 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
                   key=epoch_of)
    E = []; size_acc = []
    for f in files:
        n = ok = 0
        for m in BLOCK.finditer(open(f).read()):
            a0 = int(m.group('a0')); a1 = int(m.group('a1'))
            g0 = int(m.group('g0')); g1 = int(m.group('g1'))
            n += 1
            same_sign = (a0 > 0) == (g0 > 0)
            if a1 and g1 and same_sign and abs(math.log(abs(g1)) - math.log(abs(a1))) < logA:
                ok += 1
        E.append(epoch_of(f))
        size_acc.append(float(100.0 * ok / n) if n else float('nan'))
    o = sorted(range(len(E)), key=lambda i: E[i])
    return [int(E[i]) for i in o], [size_acc[i] for i in o]

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9.4, 5.6))
for lab, folder, col, ls, lw in RUNS:
    E, sz = curve(folder)
    pk = max(v for v in sz if v == v)
    ax.plot(E, sz, color=col, linestyle=ls, lw=float(lw),
            label=lab + '  (peak %.1f%%)' % pk)
ax.set_xlabel('epoch')
ax.set_ylabel(r'size accuracy on $r_k$   (%)')
ax.set_title(r'Learning $r_k$: same $k,\omega$ + extra features')
ax.grid(True, color='0.92'); ax.set_axisbelow(True)
ax.set_ylim(int(YMIN), int(101))
ax.set_xlim(int(0), None)
ax.legend(loc='lower right', fontsize=int(9))
#ax.text(0.014, 0.03, 'correct: same sign and  |r_k|/2 < |s_k| < 2|r_k|',
#        transform=ax.transAxes, fontsize=float(8.5), color='0.35')
fig.tight_layout()
plt.show()
#fig.savefig('cmp_PB_3_6_67.png', dpi=int(140))
#fig.savefig('cmp_PB_3_6_67.pdf')
#print('saved cmp_PB_3_6_67.png / .pdf')


# In[ ]:




