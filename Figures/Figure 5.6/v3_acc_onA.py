#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# plot_acc_v3_three_A.sage
#
# exp 19.2 (p=3, factor(Delta)):  P(model correct | true v_3(n_k) = g) vs epoch,
# grouped into g = 0, 1, >1, but RESTRICTED to k in the set
#     A = { k : 3 does not divide k,  and  k*(k^2-16) is squarefree }.
# (Since k^2-16 squarefree forces k odd and coprime to k^2-16, this is
#  k odd, 3 nmid k, and k, k-4, k+4 all squarefree.)
#   v_3(n_k) = -(first entry of answer / greedy decoded).
#
# Run:  sage plot_acc_v3_three_A.sage
import os, re, glob
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def epoch_of(p): return int(p.rsplit('.', 1)[1])
files = []
for c in [os.path.join(DATA_DIR, '../MahlerExperiments/v3/v3_fD/v3_fD run 2 (best run)'), DATA_DIR]:
    files = sorted(glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')), key=epoch_of)
    if files: break

def in_A(k):
    k = Integer(k)
    if k % 2 == 0 or k % 3 == 0:
        return False
    return k.is_squarefree() and (k - 4).is_squarefree() and (k + 4).is_squarefree()

BLOCK = re.compile(r"'k':\s*(?P<k>\d+).*?answer=\[(?P<t>-?\d+),.*?greedy\s*\n\s*decoded=\[(?P<p>-?\d+)", re.S)
def group(t): return int(t) if t <= 1 else 2      # 0, 1, or 2 (= ">1")

memb = {}
eps, s0, s1, s2 = [], [], [], []
for f in files:
    acc = {0: [0, 0], 1: [0, 0], 2: [0, 0]}
    for m in BLOCK.finditer(open(f).read()):
        k = int(m.group('k'))
        if k not in memb:
            memb[k] = in_A(k)
        if not memb[k]:
            continue
        t = -int(m.group('t')); p = -int(m.group('p')); g = group(t)
        acc[g][1] += 1
        acc[g][0] += int(t == p)
    eps.append(epoch_of(f))
    s0.append(float(100.0 * acc[0][0] / acc[0][1]) if acc[0][1] else float('nan'))
    s1.append(float(100.0 * acc[1][0] / acc[1][1]) if acc[1][1] else float('nan'))
    s2.append(float(100.0 * acc[2][0] / acc[2][1]) if acc[2][1] else float('nan'))
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
ax.set_title(r'Accuracy conditioned on $v_3(n_k)$, restricted to $k\in A$')
ax.grid(True, color='0.9')
ax.legend(loc='center right')
ax.set_ylim(int(-2), int(103))
fig.tight_layout()
#fig.savefig('acc_v3_three_A.png', dpi=int(140))
#fig.savefig('acc_v3_three_A.pdf')
#print("saved acc_v3_three_A.png / .pdf")
plt.show()


# In[ ]:




