#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pearson_pred_vs_Phat.sage
#
# Pearson correlation coefficient r between the MODEL's predicted valuation
# v_2(n_k)^{model}  and the deterministic predictor
#        P_hat(k) := round( v_hat(k) + s(k) + 1 )
# over the 3000 validation k, as a function of the epoch.
#
#   v_hat(k) = omega_odd(k) + 2*(omega_odd(k-4) + omega_odd(k+4)) - c(k)
#   c(k)     = 4 if (k odd or 16 | k)  else 2
#   s(k)     = (3/2) * max(1 - 2^(5 - M), 0),   M = max(v_2(k-8), v_2(k+8))
#   model prediction = | first entry of the greedy-decoded answer |
#
# Run:  sage pearson_pred_vs_Phat.sage
import os, re, glob
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def epoch_of(path):
    return int(path.rsplit('.', 1)[1])
files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, '../MahlerExperiments/v2/v2_fD/v2_fD run 1 (best run)/eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
               key=epoch_of)
if not files:  # also works when run from inside the 'v2_fD run 1 (best run)' folder
    files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
                   key=epoch_of)
if not files:
    raise FileNotFoundError('no eval.valid.elliptic_curve.* files found -- '
        'expected the MahlerExperiments folder inside the Figures folder '
        '(see the path at the top of this script)')
# ---- arithmetic ----
def omega_odd(m):
    m = ZZ(m).abs()
    if m == 0:
        return 0
    return sum(1 for p in m.prime_divisors() if p != 2)
def vhat(k):
    k = ZZ(k)
    c = 4 if (k % 2 == 1 or k % 16 == 0) else 2
    return omega_odd(k) + 2 * (omega_odd(k - 4) + omega_odd(k + 4)) - c
def s_func(k):
    k = ZZ(k)
    M = max((k - 8).valuation(2), (k + 8).valuation(2))
    return QQ(3) / 2 * max(1 - 2 ^ (5 - M), 0)
def P_hat(k):
    # deterministic predictor: round( v_hat(k) + s(k) + 1 )
    return float(round(float(vhat(k) + s_func(k) + 1)))
# ---- parse one epoch dump -> lists (pred_v2, P_hat) ----
BLOCK = re.compile(r"'k':\s*(?P<k>\d+).*?greedy\s*\n\s*decoded=\[(?P<first>-?\d+)", re.S)
def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return float(sxy / (sxx * syy) ** 0.5)
def load(path):
    preds, base = [], []
    for m in BLOCK.finditer(open(path).read()):
        k = ZZ(m.group('k'))
        preds.append(float(abs(ZZ(m.group('first')))))   # v_2(n_k)^model
        base.append(P_hat(k))                             # P_hat(k)
    return preds, base
# ---- Pearson r for every epoch ----
points = []
for f in files:
    preds, base = load(f)
    points.append((epoch_of(f), pearson(preds, base)))
points.sort()
print("epochs parsed: %d  (n=%d validation k)" % (len(points), len(preds)))
print("final epoch %d : Pearson r( model prediction , P_hat ) = %.4f"
      % (points[-1][0], points[-1][1]))
# ---- plot vs epoch (matplotlib bundled with Sage; PDF-safe) ----
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
xs = [int(p[0]) for p in points]
ys = [float(p[1]) for p in points]
fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.plot(xs, ys, color='crimson', lw=float(1.5), marker='o', ms=float(3),
        label=r'$r(\,\mathrm{model\ prediction},\ \hat{P}(k)\,)$')
ax.set_xlabel('epoch')
ax.set_ylabel('Pearson correlation  r')
ax.set_title(r'Correlation of model predictions with $\hat{P}(k)=\mathrm{round}(\hat{v}(k)+s(k)+1)$ vs. epoch')
ax.grid(True, color='0.85')                 # solid gray, no alpha -> PDF-safe
ax.legend(loc='lower right', framealpha=1.0)
fig.tight_layout()
#fig.savefig('pearson_pred_vs_Phat_vs_epoch.png', dpi=int(150))
#fig.savefig('pearson_pred_vs_Phat_vs_epoch.pdf')
#print("saved pearson_pred_vs_Phat_vs_epoch.png / .pdf")
plt.show()


# In[ ]:




