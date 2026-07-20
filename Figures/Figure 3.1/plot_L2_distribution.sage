# Figure 3.1 -- Distribution of L(E_k, 2) = 4*pi^2 * |n_k| * mu_k / N_k  over all k
# with 1 <= k <= 250000.
#
#   * |n_k| = 1/|r_k|  (from r_k = Fraction(num, den) in output_full.txt);
#   * mu_k  = Mahler measure of  x+1/x+y+1/y+k  (hypergeometric series for |k|>4,
#             direct double integral for k = 1,2,3);
#   * N_k   = conductor of E_k  (field [3][0] of output_full.txt).
# k = 4 is skipped (k^2-16 = 0, degenerate), giving n = 249999 as in the paper.
# Values fall in a narrow O(1) range (min 0.5222, mean 0.8326, max 1.7532).
#
# Displays the figure (plt.show); nothing is written to disk.
# Run:  sage plot_L2_distribution.sage       (or in a notebook / python3)
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

LO, HI, NBINS = 0.40, 1.80, 80

def mu_hyp(k):                                   # |k| > 4: hypergeometric series (z = 16/k^2 < 1)
    k = abs(float(k)); z = 16.0 / k / k; term = 1.0; F = 1.0; n = 0
    while abs(term) > 1e-17 and n < 300:
        term *= (1.5 + n) ** 2 * (n + 1) * z / ((2.0 + n) ** 3); F += term; n += 1
    return math.log(k) - (2.0 / (k * k)) * F

def mu_num(k):                                   # |k| <= 4: direct Mahler-measure integral
    g = 800
    c = 2.0 * np.cos((np.arange(g) + 0.5) / g * 2.0 * math.pi)
    tot = 0.0
    for cv in c:
        tot += np.mean(np.log(np.abs(k + c + cv)))
    return tot / g

def mu(k):
    return mu_hyp(k) if abs(k) > 4 else mu_num(k)

frac = re.compile(r'Fraction\((-?\d+),\s*(-?\d+)\)')
BIG  = re.compile(r'^\[(\d+), (-?\d+), \[Fraction\([^)]*\), \[\[.*?\]\]\], \[(\d+),')
L = []
with open(OUTPUT_FULL) as fh:
    for line in fh:
        if not line.startswith('['):
            continue
        m = BIG.match(line)
        if not m:
            continue
        k = int(m.group(1))
        if not (1 <= k <= 250000) or k == 4:        # k = 4 is degenerate (k^2-16 = 0)
            continue
        N = int(m.group(3))
        num, den = map(int, frac.findall(line)[-1])          # r_k = num/den
        inv = abs(den) / abs(num)                            # |n_k| = |1/r_k|
        L.append(4.0 * math.pi ** 2 * mu(k) * inv / N)
L = np.array([x for x in L if math.isfinite(x)])
print('n = %d   min=%.4f  mean=%.4f  max=%.4f' % (len(L), L.min(), L.mean(), L.max()))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(float(7.2), float(4.6)))
ax.hist(L, bins=int(NBINS), range=(float(LO), float(HI)), color='#1f77b4')
ax.set_xlabel(r'$L(E_k,2)=4\pi^{2}\,|n_k|\,\mu_k/N_k$')
ax.set_ylabel('count')
ax.set_title(r'Distribution of $L(E_k,2)$ over all $k$   ($n = %d$)' % len(L))
ax.set_xlim(float(LO), float(HI))
fig.tight_layout()
plt.show()