# Figure 3.3 -- v_2(n_k) vs the approximation vhat(k), for k with v_2(n_k) >= 0.
#
#   vhat(k) = omega_odd(k) + 2*(omega_odd(k-4) + omega_odd(k+4)) - c(k),
#   where omega_odd(n) = # distinct odd prime divisors of n, and
#   c(k) = 4 if E_k is semistable at 2 (k odd or 16 | k), else 2.
#   v_2(n_k) is read from the data (n_k = 1/r_k, r_k = Fraction(num,den)).
#
# Left panel : mean v_2 per vhat-bin, the least-squares regression line, and y = x.
# Right panel: 2D histogram of (vhat, v_2) with the y = x diagonal and a colorbar.
#
# Displays the figure (plt.show); nothing is written to disk.
# Run:  sage plot_v2_vs_vhat.sage      (or in a notebook / python3)
import os, re, glob, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def find(name):
    for c in [os.path.join(HERE, name), name]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(HERE, '**', name), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(name)
OUTPUT_FULL = find('output_full.txt')

# ---- smallest-prime-factor sieve for omega_odd ----
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
        p = int(spf[n])
        if p != 2:
            s += 1
        while n % p == 0:
            n //= p
    return s
def c_of(k):
    return 4 if (k % 2 == 1 or k % 16 == 0) else 2
def vhat(k):
    return omega_odd(k) + 2 * (omega_odd(k - 4) + omega_odd(k + 4)) - c_of(k)

def v2(n):
    n = abs(int(n)); v = 0
    if n == 0:
        return 0
    while n % 2 == 0:
        n //= 2; v += 1
    return v

frac = re.compile(r'Fraction\((-?\d+),\s*(-?\d+)\)')
VH, V2 = [], []
with open(OUTPUT_FULL) as fh:
    for line in fh:
        if not line.startswith('['):
            continue
        k = int(line[1:line.index(',')])
        if not (1 <= k <= 250000) or k == 4:
            continue
        num, den = map(int, frac.findall(line)[-1])
        v = v2(den) - v2(num)                       # v_2(n_k)
        if v < 0:                                   # keep k with integer n_k (v_2 >= 0)
            continue
        VH.append(vhat(k)); V2.append(v)
VH = np.array(VH); V2 = np.array(V2)

slope, intercept = np.polyfit(VH, V2, 1)
r = float(np.corrcoef(VH, V2)[0, 1])
print('n=%d  slope=%.2f  intercept=%.2f  r=%.3f' % (len(VH), slope, intercept, r))

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(float(11.0), float(4.6)))

# left: mean v_2 per vhat bin + regression + y=x
xs = np.arange(int(VH.min()), int(VH.max()) + 1)
means = [V2[VH == x].mean() for x in xs]
ax1.scatter(xs, means, s=int(14), color='#1f77b4', label=r'mean $v_2$ per bin', zorder=int(3))
xx = np.array([VH.min(), VH.max()], dtype=float)
ax1.plot(xx, slope * xx + intercept, color='#d62728', lw=float(1.5),
         label=r'regression: $v_2=%.2f\hat v+%.2f$ ($r=%.3f$)' % (slope, intercept, r))
ax1.plot(xx, xx, color='0.5', ls='--', lw=float(1.0), label=r'$v_2=\hat v$')
ax1.set_xlabel(r'approximation $\hat v$'); ax1.set_ylabel(r'$v_2(n_k)$')
ax1.legend(loc='upper left', fontsize=int(8))

# right: 2D histogram
xb = np.arange(VH.min() - 0.5, VH.max() + 1.5)
yb = np.arange(-0.5, V2.max() + 1.5)
h = ax2.hist2d(VH, V2, bins=[xb, yb], cmap='viridis')
fig.colorbar(h[3], ax=ax2, label='frequency')
ax2.plot([0, min(VH.max(), V2.max())], [0, min(VH.max(), V2.max())], color='white', lw=float(1.0))
ax2.set_xlabel(r'approximation $\hat v$'); ax2.set_ylabel(r'actual $v_2(n_k)$')
ax2.grid(True, color='white', alpha=float(0.15))
fig.tight_layout()
plt.show()
