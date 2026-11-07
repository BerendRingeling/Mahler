# Figure 6.1 -- Histogram of the 3-adic valuations v_3(n_k) for 1 <= k <= 250K.
#
# Data source:  output_full.txt  (r_k = Fraction(num, den);  n_k = 1/r_k, so
#               v_3(n_k) = v_3(den) - v_3(num)).
#
# Displays the figure (plt.show); nothing is written to disk.
# Run:  sage plot_v3_histogram.sage      (or notebook / python3)
import os, re, glob
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
OUTPUT_FULL = find('../MahlerExperiments/data/output_full.txt')

def vp(n, p):
    n = abs(int(n)); v = 0
    if n == 0:
        return 0
    while n % p == 0:
        n //= p; v += 1
    return v

frac = re.compile(r'Fraction\((-?\d+),\s*(-?\d+)\)')
vals = []
with open(OUTPUT_FULL) as fh:
    for line in fh:
        if not line.startswith('['):
            continue
        k = int(line[1:line.index(',')])
        if not (1 <= k <= 250000):
            continue
        num, den = map(int, frac.findall(line)[-1])
        vals.append(vp(den, 3) - vp(num, 3))
vals = np.array(vals)
V = int(vals.max())
counts = np.array([int(np.sum(vals == r)) for r in range(V + 1)])
print('n = %d   max v_3 = %d' % (len(vals), V))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(float(7.4), float(4.5)))
ax.bar(range(V + 1), counts, width=float(0.85), color='#1f77b4')
ax.set_xlabel(r'$v_3(n_k)$')
ax.set_ylabel('count')
ax.set_title(r'Histogram of all non-negative $v_3(n_k)$ for $1 \leq k \leq 250K$')
ax.set_xticks(range(V + 1))
ax.tick_params(axis='x', labelsize=int(8))
fig.tight_layout()
plt.show()
