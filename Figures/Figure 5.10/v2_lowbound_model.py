#!/usr/bin/env python
# coding: utf-8

# In[2]:


# plot_lowerbound_proportion.sage
#
# For each training epoch, computes the proportion (out of the 3000 validation k)
# for which the MODEL's predicted valuation satisfies the conjectural lower bound
#         v_hat(k) < v_2(n_k)^{model}
# and plots that proportion as a function of epoch.
#
#   v_hat(k) = omega_odd(k) + 2*omega_odd(k^2-16) - c(k),
#   c(k)     = 4 if (k odd or 16 | k)   [semistable at 2]
#              2 otherwise               [additive at 2]
# and omega_odd(k^2-16) = omega_odd(k-4) + omega_odd(k+4)
# (no odd prime divides both k-4 and k+4, since their difference is 8).
#
# The model's predicted v_2(n_k) is |first entry of the greedy-decoded answer|.
#
# Run:  sage plot_lowerbound_proportion.sage
# Output: lowerbound_proportion_vs_epoch.png / .pdf

import os, re, glob

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
MAX_EPOCH = 20


def epoch_of(path):
    return int(path.rsplit('.', 1)[1])

files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, '../MahlerExperiments/v2/v2_fD/v2_fD run 1 (best run)/eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
               key=epoch_of)
if not files:  # allow running from inside the 'v2_fD run 1 (best run)' folder too
    files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
                   key=epoch_of)
if not files:
    raise FileNotFoundError('no eval.valid.elliptic_curve.* files found -- '
        'expected the MahlerExperiments folder inside the Figures folder '
        '(see the path at the top of this script)')

# ---- v_hat(k) ----
def omega_odd(m):
    m = ZZ(m).abs()
    if m == 0:
        return 0
    return sum(1 for p in m.prime_divisors() if p != 2)

def vhat(k):
    k = ZZ(k)
    c = 4 if (k % 2 == 1 or k % 16 == 0) else 2
    return omega_odd(k) + 2 * (omega_odd(k - 4) + omega_odd(k + 4)) - c

# ---- parse one epoch dump: yield (k, predicted_v2) ----
BLOCK = re.compile(
    r"'k':\s*(?P<k>\d+).*?greedy\s*\n\s*decoded=\[(?P<first>-?\d+)",
    re.S,
)

def proportion(path):
    hits = total = 0
    for m in BLOCK.finditer(open(path).read()):
        k = int(m.group('k'))
        pred_v2 = abs(int(m.group('first')))   # model's predicted v_2(n_k)
        total += 1
        if vhat(k) <= pred_v2:                  # strict conjectural lower bound
            hits += 1
    return (float(100.0 * hits / total) if total else None), total

points = []
for f in files:
    prop, tot = proportion(f)
    if prop is not None:
        points.append((epoch_of(f), prop))
points.sort()

points = [p for p in points if p[0] <= MAX_EPOCH]


final_ep, final_prop = points[-1]
print("parsed %d epochs (%d validation k)" % (len(points), tot))
print("final model (epoch %d): v_hat(k) <= v_2(n_k)^model for %.1f%% of k"
      % (final_ep, final_prop))

# ---- plot (matplotlib bundled with Sage; PDF-safe: no alpha, plain-python numbers) ----
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

xs = [int(p[0]) for p in points]
ys = [float(p[1]) for p in points]

fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.plot(xs, ys, color='darkgreen', lw=float(1.5), marker='o', ms=float(3),
        label=r'proportion with $\hat{v}(k) \leq v_2(n_k)$ (model)')
ax.set_xlabel('epoch')
ax.set_ylabel('proportion of validation k  (%)')
ax.set_title(r'Fraction of the 3000 validation $k$ where the model output '
             r'satisfies the lower bound $\hat{v}(k) \leq v_2(n_k)$')
ax.grid(True, color='0.85')                    # solid gray, no alpha -> PDF-safe
ax.legend(loc='lower right', framealpha=1.0)
from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
fig.tight_layout()
#fig.savefig('lowerbound_proportion_vs_epoch.png', dpi=int(150))
#fig.savefig('lowerbound_proportion_vs_epoch.pdf')
#print("saved lowerbound_proportion_vs_epoch.png / .pdf")
plt.show()


# In[ ]:




