#!/usr/bin/env python
# coding: utf-8

# In[2]:


# plot_correct_given_G.sage
#
# P(model correct | G = g) as a function of the epoch, for g = 0, 1, 2, 3,
# over the 3000 validation k, where the excess is
#        G(k) = v_2(n_k) - v_hat(k).
# G = 0 are the k sitting exactly on the lower bound v_hat(k).
#
#   v_hat(k) = omega_odd(k) + 2*(omega_odd(k-4) + omega_odd(k+4)) - c(k)
#   c(k)     = 4 if (k odd or 16 | k)  else 2
#   true v_2(n_k) = |first entry of answer|,  model = |first entry of greedy|
#
# Run:  sage plot_correct_given_G.sage
import os, re, glob
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
G_VALUES = [0]
def epoch_of(path):
    return int(path.rsplit('.', 1)[1])
files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, '../MahlerExperiments/v2/v2_fD/v2_fD run 3 (best run)/eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
               key=epoch_of)
if not files:
    files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
                   key=epoch_of)
if not files:
    raise FileNotFoundError('no eval.valid.elliptic_curve.* files found -- '
        'expected the MahlerExperiments folder inside the Figures folder '
        '(see the path at the top of this script)')
# ---- v_hat(k)  (cached per k) ----
def omega_odd(m):
    m = ZZ(m).abs()
    return 0 if m == 0 else sum(1 for p in m.prime_divisors() if p != 2)
_base = {}
def base(k):
    k = ZZ(k)
    if k not in _base:
        c = 4 if (k % 2 == 1 or k % 16 == 0) else 2
        vhat = omega_odd(k) + 2 * (omega_odd(k - 4) + omega_odd(k + 4)) - c
        _base[k] = float(vhat)
    return _base[k]
# ---- parse: for each epoch, correct/total counts split by G ----
BLOCK = re.compile(r"'k':\s*(?P<k>\d+).*?answer=\[(?P<t>-?\d+).*?greedy\s*\n\s*decoded=\[(?P<p>-?\d+)", re.S)
def counts(path):
    corr = {g: 0 for g in G_VALUES}
    tot  = {g: 0 for g in G_VALUES}
    for m in BLOCK.finditer(open(path).read()):
        k    = ZZ(m.group('k'))
        v2   = abs(int(m.group('t')))
        pred = abs(int(m.group('p')))
        G = int(v2 - base(k))          # G(k) = v_2(n_k) - v_hat(k)
        if G in tot:
            tot[G] += 1
            if pred == v2:
                corr[G] += 1
    return corr, tot
# ---- per epoch, per G ----
series = {g: [] for g in G_VALUES}
for f in files:
    corr, tot = counts(f)
    e = epoch_of(f)
    for g in G_VALUES:
        if tot[g] > 0:
            series[g].append((e, float(100.0 * corr[g] / tot[g])))
for g in G_VALUES:
    series[g].sort()
    print("G = %d : final P(correct) = %.1f%%   (n = %d)" % (g, series[g][-1][1], tot[g]))
# ---- plot (matplotlib bundled with Sage; PDF-safe) ----
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
colors = {0: 'green', 1: 'steelblue', 2: 'darkorange', 3: 'crimson'}
fig, ax = plt.subplots(figsize=(9.0, 5.0))
for g in G_VALUES:
    xs = [int(p[0]) for p in series[g]]
    ys = [float(p[1]) for p in series[g]]
    ax.plot(xs, ys, color=colors[g], lw=float(1.4), marker='o', ms=float(2.5),
            label=r'$G = %d$' % g)
ax.set_xlabel('epoch')
ax.set_ylabel('P(model correct | G(k) =  0)   (%)')
ax.set_title(r'Accuracy by excess $G(k) = v_2(n_k) - \hat{v}(k)$ vs. epoch')
ax.grid(True, color='0.85')                 # solid gray, no alpha -> PDF-safe
ax.legend(loc='upper left', framealpha=1.0, title='excess')
fig.tight_layout()
#fig.savefig('correct_given_G_vs_epoch.png', dpi=int(150))
#fig.savefig('correct_given_G_vs_epoch.pdf')
#print("saved correct_given_G_vs_epoch.png / .pdf")
plt.show()


# In[ ]:




