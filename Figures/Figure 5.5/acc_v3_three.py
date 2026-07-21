#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# plot_exp19_acc_v3_three.sage
#
# For the p=3 factor(D) run ('v3_fD run 2 (best run)'): P(model correct | true v_3) vs epoch,
# grouped into three curves:  v_3 = 0,  v_3 = 1,  v_3 > 1.
#   v_3(n_k) = -(first entry of answer/greedy).
#
# Run:  sage plot_exp19_acc_v3_three.sage
import os, re, glob
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def epoch_of(p): return int(p.rsplit('.', 1)[1])
files = []
for c in [os.path.join(DATA_DIR, '../MahlerExperiments/v3/v3_fD/v3_fD run 2 (best run)'), DATA_DIR]:
    files = sorted(glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')), key=epoch_of)
    if files: break

BLOCK = re.compile(r"answer=\[(?P<t>-?\d+).*?greedy\s*\n\s*decoded=\[(?P<p>-?\d+)", re.S)
def group(t):            # 0, 1, or 2 (=">1")
    return t if t <= 1 else 2

eps, s0, s1, s2 = [], [], [], []
for f in files:
    T, P = [], []
    for m in BLOCK.finditer(open(f).read()):
        T.append(-int(m.group('t'))); P.append(-int(m.group('p')))
    eps.append(epoch_of(f))
    def cond(g):
        num = sum(1 for t, p in zip(T, P) if group(t) == g and p == t)
        den = sum(1 for t in T if group(t) == g)
        return float(100.0 * num / den) if den else float('nan')
    s0.append(cond(0)); s1.append(cond(1)); s2.append(cond(2))
o = sorted(range(len(eps)), key=lambda i: eps[i])
eps = [int(eps[i]) for i in o]
s0 = [s0[i] for i in o]; s1 = [s1[i] for i in o]; s2 = [s2[i] for i in o]

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.plot(eps, s1, color='#2ca02c', lw=float(1.6), label=r'$v_3 = 1$')
ax.plot(eps, s0, color='#1f77b4', lw=float(1.6), label=r'$v_3 = 0$')
ax.plot(eps, s2, color='#d62728', lw=float(1.6), label=r'$v_3 > 1$')
ax.set_xlabel('epoch')
ax.set_ylabel(r'$P(\mathrm{model\ correct}\mid v_3)$   (%)')
ax.set_title(r'Accuracy conditioned on $v_3(n_k)$')
ax.grid(True, color='0.9')
ax.legend(loc='center right')
ax.set_ylim(-2, 103)
fig.tight_layout()
#fig.savefig('exp19_acc_v3_three.png', dpi=int(140))
#fig.savefig('exp19_acc_v3_three.pdf')
#print("saved exp19_acc_v3_three.png / .pdf")
plt.show()


# In[ ]:




