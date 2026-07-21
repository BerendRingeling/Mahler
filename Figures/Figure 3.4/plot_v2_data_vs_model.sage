# Figure 3.4 -- Observed v_2(n_k) (for k with v_2(n_k) >= 0) against the model
#   v_2(n_k) ~ vhat(k) + s(k) + X,   X ~ NB(r=3, p=3/5)   (mean 2, mode 1),
# where
#   vhat(k) = omega_odd(k) + 2(omega_odd(k-4)+omega_odd(k+4)) - c(k),
#             c(k) = 4 if (k odd or 16|k) else 2,
#   s(k)    = (3/2) * max(1 - 2^(5-M), 0),  M = max(v_2(k-8), v_2(k+8)).
# The model histogram convolves the distribution of round(vhat+s) with the NB pmf.
#
# Displays a grouped bar chart (observed vs model); nothing is written to disk.
# Run:  sage plot_v2_data_vs_model.sage      (or notebook / python3)
import os, re, glob, math
import numpy as np
from collections import Counter
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def find(name):
    for c in [os.path.join(HERE, name), name]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(HERE, '**', name), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(name)
OUTPUT_FULL = find('../MahlerExperiments/data/output_full.txt')

# ---- omega_odd via smallest-prime-factor sieve ----
NMAX = 250010
spf = np.arange(NMAX)
for i in range(2, int(NMAX ** 0.5) + 1):
    if spf[i] == i:
        spf[i * i::i] = np.minimum(spf[i * i::i], i)
def omega_odd(n):
    n = abs(int(n))
    if n <= 1:
        return 0
    s = 0
    while n > 1:
        pp = int(spf[n])
        if pp != 2:
            s += 1
        while n % pp == 0:
            n //= pp
    return s
def v2i(n):
    n = abs(int(n))
    if n == 0:
        return 0
    v = 0
    while n % 2 == 0:
        n //= 2; v += 1
    return v
def vhat(k):
    c = 4 if (k % 2 == 1 or k % 16 == 0) else 2
    return omega_odd(k) + 2 * (omega_odd(k - 4) + omega_odd(k + 4)) - c
def s_of(k):
    M = max(v2i(k - 8), v2i(k + 8))
    return 1.5 * max(1.0 - 2.0 ** (5 - M), 0.0)

# ---- observed v_2(n_k) and base = round(vhat+s) ----
frac = re.compile(r'Fraction\((-?\d+),\s*(-?\d+)\)')
obs = Counter(); base = Counter(); N = 0
with open(OUTPUT_FULL) as fh:
    for line in fh:
        if not line.startswith('['):
            continue
        k = int(line[1:line.index(',')])
        if not (1 <= k <= 250000) or k == 4:
            continue
        num, den = map(int, frac.findall(line)[-1])
        v = v2i(den) - v2i(num)                      # v_2(n_k)
        if v < 0:
            continue
        obs[v] += 1
        base[int(round(vhat(k) + s_of(k)))] += 1
        N += 1

# ---- NB(3, 3/5) pmf and model histogram = base * NB ----
rr, pp = 3, 3.0 / 5.0
def nb(x):
    return comb(x + rr - 1, x) * pp ** rr * (1 - pp) ** x if x >= 0 else 0.0
VMAX = max(max(obs), max(base) + 30)
model = np.zeros(VMAX + 1)
for b, cnt in base.items():
    for j in range(0, VMAX - b + 1):
        model[b + j] += cnt * nb(j)
observed = np.array([obs.get(v, 0) for v in range(VMAX + 1)], dtype=float)

# ---- chi^2 / nu over bins with a meaningful model count ----
use = model >= 5
chi2 = float(np.sum((observed[use] - model[use]) ** 2 / model[use]))
nu = int(np.sum(use)) - 1
print('N=%d  chi2=%.1f  nu=%d  chi2/nu=%.2f' % (N, chi2, nu, chi2 / nu))

import matplotlib.pyplot as plt
x = np.arange(VMAX + 1)
w = 0.4
fig, ax = plt.subplots(figsize=(float(8.2), float(4.8)))
ax.bar(x - w / 2, observed, width=float(w), color='#1f77b4', label=r'observed $v_2(n_k)$')
ax.bar(x + w / 2, model, width=float(w), color='#ff7f0e',
       label=r'model $\hat v(k)+s(k)+\mathrm{NB}(3,\frac{3}{5})$')
ax.set_xlabel(r'$v_2(n_k)$'); ax.set_ylabel('count')
ax.set_title(r'$v_2(n_k)$: data vs model   ($N = %s$, $\chi^2/\nu = %.1f$)'
             % ('{:,}'.format(N), chi2 / nu))
ax.set_xlim(-1, min(VMAX, 27))
ax.set_xticks(range(0, min(VMAX, 27) + 1, 2))
ax.legend(loc='upper right')
fig.tight_layout()
plt.show()
