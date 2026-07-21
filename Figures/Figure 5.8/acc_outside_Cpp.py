#!/usr/bin/env python
# coding: utf-8

# In[3]:


# plot_acc_outside_Cpp.sage
#
# exp 19.2 (p=3, factor(Delta)): model floor-detection accuracy on
# v_3(n_k) in {0,1}, restricted to the complement of
#      C'' u { k = 0, +-5  (mod 27) },
# as a function of epoch, together with the guess-1 baseline.
# Here
#   C''  =  { p^2 nmid k^2-16  for all p = 3 (mod 4) }        (squarefree at 3 mod 4)
#           \ { k = 4 (mod 8) }                               (the hat: remove T)
#           n  { p^3 | k(k^2-16) => p = 2 (mod 3) }           (the star: cube condition)
# i.e. C'' = C' n {cube condition on k and on the odd part of k^2-16}.
#   v_3(n_k) = -(first entry of answer / greedy decoded).
#
# Run:  sage plot_acc_outside_Cpp.sage
import os, re, glob
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def epoch_of(p): return int(p.rsplit('.', 1)[1])
files = []
for c in [os.path.join(DATA_DIR, '../MahlerExperiments/v3/v3_fD/v3_fD run 2 (best run)'), DATA_DIR]:
    files = sorted([f for f in glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()], key=epoch_of)
    if files: break
if not files:
    raise FileNotFoundError('no eval.valid.elliptic_curve.* files found -- '
        'expected the MahlerExperiments folder inside the Figures folder '
        '(see the path at the top of this script)')

# ---- membership ----
def factor_dict(n):
    d = {}
    for p, e in Integer(abs(n)).factor():
        d[p] = d.get(p, 0) + e
    return d
def factorD(k):                       # factorisation of k^2-16 = (k-4)(k+4)
    d = factor_dict(k - 4)
    for p, e in factor_dict(k + 4).items():
        d[p] = d.get(p, 0) + e
    return d
def in_Cdoubleprime(k):               # C''
    fk = factor_dict(k); fD = factorD(k)
    # C': squarefree at all p = 3 mod 4, and k != 4 mod 8
    if any(p != 2 and p % 4 == 3 and e >= 2 for p, e in fD.items()):
        return False
    if valuation(k, 2) == 2:
        return False
    # star (cube condition): p^3 | k(k^2-16) => p = 2 mod 3, on k and odd part of k^2-16
    if not all((e < 3 or p % 3 == 2) for p, e in fk.items()):
        return False
    if not all((e < 3 or p % 3 == 2) for p, e in fD.items() if p != 2):
        return False
    return True
def target(k):                        # complement of  C'' u {k = 0,+-5 mod 27}
    return (not in_Cdoubleprime(k)) and (k % 27 not in (0, 5, 22))

BLOCK = re.compile(r"'k':\s*(?P<k>\d+).*?answer=\[(?P<t>-?\d+),.*?greedy\s*\n\s*decoded=\[(?P<p>-?\d+)", re.S)
memb = {}
E = []; acc = []; base = []
for f in files:
    ok = tot = b = 0
    for m in BLOCK.finditer(open(f).read()):
        k = int(m.group('k'))
        if k not in memb:
            memb[k] = target(k)
        if not memb[k]:
            continue
        t = -int(m.group('t'))
        if not (0 <= t <= 1):
            continue
        p = -int(m.group('p'))
        tot += 1; ok += int(t == p); b += int(t == 1)
    E.append(epoch_of(f))
    acc.append(float(100.0 * ok / tot) if tot else float('nan'))
    base.append(float(100.0 * b / tot) if tot else float('nan'))
o = sorted(range(len(E)), key=lambda i: E[i])
E = [int(E[i]) for i in o]; acc = [acc[i] for i in o]; base = [base[i] for i in o]

def smooth(y, w=5):
    y = np.array(y, dtype=float); out = y.copy()
    for i in range(len(y)):
        out[i] = np.nanmean(y[max(0, i - w):i + 1])
    return out

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9.2, 5.2))
ax.plot(E, smooth(acc), color='#8c564b', lw=float(2.0),
        label=r"model on complement of $C''\cup\{k\equiv0,\pm5\,(27)\}$")
#ax.axhline(float(np.nanmean(base)), color='#999999', lw=float(1.3), ls='--',
#           label=r'baseline (guess $v_3=1$)')
ax.set_xlabel('epoch')
ax.set_ylabel(r'accuracy on $v_3(n_k)\in\{0,1\}$   (%)')
ax.set_title(r"Accuracy outside $C''$ and the mod-27 classes")
ax.grid(True, color='0.92')
ax.set_ylim(int(40), int(100))
ax.legend(loc='lower right')
fig.tight_layout()
#fig.savefig('acc_outside_Cpp.png', dpi=int(140))
#fig.savefig('acc_outside_Cpp.pdf')
#print("saved acc_outside_Cpp.png / .pdf")
plt.show()


# In[ ]:




