#!/usr/bin/env python
# coding: utf-8

# In[1]:


# plot_acc_by_set.sage
#
# exp 19.2 (p=3, factor(Delta)): model floor-detection accuracy on
# v_3(n_k) in {0,1}, as a function of epoch, for each of the nine sets
#   A, A', A''   (squarefree chain)
#   B, B', B''   (odd part of k^2-16 squarefree)
#   C, C', C''   (k^2-16 squarefree at all p = 3 mod 4)
# Colour = family (A/B/C), line style = repair level (base / ' / '').
#   v_3(n_k) = -(first entry of answer / greedy decoded).
#
# Run:  sage plot_acc_by_set.sage
import os, re, glob
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def epoch_of(p): return int(p.rsplit('.', 1)[1])
files = []
for c in [os.path.join(DATA_DIR, '../MahlerExperiments/v3/v3_fD/v3_fD run 2 (best run)'), DATA_DIR]:
    files = sorted(glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')), key=epoch_of)
    if files: break
STRIDE = 1                       # set to 3 to subsample epochs for speed
files = files[::STRIDE]

# ---- set membership for a value k ----
def factor_dict(n):
    d = {}
    for p, e in Integer(abs(n)).factor():
        d[p] = d.get(p, 0) + e
    return d
def factorD(k):                  # factorisation of k^2-16 = (k-4)(k+4)
    d = factor_dict(k - 4)
    for p, e in factor_dict(k + 4).items():
        d[p] = d.get(p, 0) + e
    return d
def membership(k):
    fk = factor_dict(k); fD = factorD(k)
    s16   = all(e == 1 for e in fD.values())                         # k^2-16 squarefree
    oddsf = all(e == 1 for p, e in fD.items() if p != 2)             # odd part squarefree
    c3    = all(not (p != 2 and p % 4 == 3 and e >= 2) for p, e in fD.items())
    no4   = (valuation(k, 2) != 2)                                   # k not = 4 mod 8
    ck    = all((e < 3 or p % 3 == 2) for p, e in fk.items())        # cube cond. on k
    ckD   = ck and all((e < 3 or p % 3 == 2) for p, e in fD.items() if p != 2)
    return {
        'A':   (k % 3 != 0 and all(e == 1 for e in fk.values()) and s16),
        "A'":  (k % 3 != 0 and s16),
        "A''": s16,
        'B':   oddsf,
        "B'":  oddsf and no4,
        "B''": oddsf and no4 and ck,
        'C':   c3,
        "C'":  c3 and no4,
        "C''": c3 and no4 and ckD,
    }

BLOCK = re.compile(r"'k':\s*(?P<k>\d+).*?answer=\[(?P<t>-?\d+),.*?greedy\s*\n\s*decoded=\[(?P<p>-?\d+)", re.S)
names = ['A', "A'", "A''", 'B', "B'", "B''", 'C', "C'", "C''"]
memb = {}
E = []; curves = {n: [] for n in names}
for f in files:
    tal = {n: [0, 0] for n in names}
    for m in BLOCK.finditer(open(f).read()):
        k = int(m.group('k')); t = -int(m.group('t'))
        if not (0 <= t <= 1):
            continue
        p = -int(m.group('p'))
        if k not in memb:
            memb[k] = membership(k)
        mm = memb[k]
        for n in names:
            if mm[n]:
                tal[n][1] += 1
                tal[n][0] += int(t == p)
    E.append(epoch_of(f))
    for n in names:
        curves[n].append(float(100.0 * tal[n][0] / tal[n][1]) if tal[n][1] else float('nan'))
o = sorted(range(len(E)), key=lambda i: E[i])
E = [int(E[i]) for i in o]
for n in names:
    curves[n] = [curves[n][i] for i in o]

def smooth(y, w=4):
    y = np.array(y, dtype=float); out = y.copy()
    for i in range(len(y)):
        out[i] = np.nanmean(y[max(0, i - w):i + 1])
    return out

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
col = {'A': '#7f7f7f', 'B': '#1f77b4', 'C': '#d62728'}
lst = {'': '-', "'": '--', "''": ':'}
fig, ax = plt.subplots(figsize=(9.6, 5.6))
for n in names:
    fam = n[0]; lvl = n[1:]
    ax.plot(E, smooth(curves[n]), color=col[fam], linestyle=lst[lvl],
            lw=float(1.7), label=r'$%s$' % n)
ax.set_xlabel('epoch')
ax.set_ylabel(r'accuracy on $v_3(n_k)\in\{0,1\}$   (%)')
ax.set_title(r'Accuracy by set vs. epoch')
ax.grid(True, color='0.92')
ax.set_ylim(int(40), int(102))
ax.legend(loc='lower right', ncol=int(3), fontsize=int(9))
fig.tight_layout()
#fig.savefig('acc_by_set_epochs.png', dpi=int(140))
#fig.savefig('acc_by_set_epochs.pdf')
#print("saved acc_by_set_epochs.png / .pdf")
plt.show()


# In[ ]:




