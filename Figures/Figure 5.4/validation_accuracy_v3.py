#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# plot_greedy_accuracy_p3.sage
# Greedy exact-match accuracy vs. epoch for the p = 3 (v_3) run ('v3_fD run 2 (best run)').
#
# Run:  sage plot_greedy_accuracy_p3.sage
# Output: greedy_accuracy_p3_vs_epoch.png / .pdf

import os, re, glob

# folder holding the p=3 dumps eval.valid.elliptic_curve.*  (edit if needed)
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
CANDIDATES = [
    os.path.join(DATA_DIR, '../MahlerExperiments/v3/v3_fD/v3_fD run 2 (best run)'),   # p = 3 run
    DATA_DIR,                                    # or run from inside that folder
]

def epoch_of(path):
    return int(path.rsplit('.', 1)[1])

files = []
for c in CANDIDATES:
    files = sorted(glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')), key=epoch_of)
    if files:
        break

# exact-match of the FULL decoded list vs answer (the greedy_acc metric)
BLOCK = re.compile(
    r"answer=\[(?P<ans>-?\d+(?:,\s*-?\d+)*)\].*?greedy\s*\n\s*decoded=\[(?P<pred>-?\d+(?:,\s*-?\d+)*)\]",
    re.S,
)
def parse_ints(s):
    return tuple(int(x) for x in s.split(','))

def greedy_accuracy(path):
    txt = open(path).read()
    n = correct = 0
    for m in BLOCK.finditer(txt):
        n += 1
        if parse_ints(m.group('ans')) == parse_ints(m.group('pred')):
            correct += 1
    return (float(100.0 * correct / n) if n else None), n

points = []
for f in files:
    acc, n = greedy_accuracy(f)
    if acc is not None:
        points.append((epoch_of(f), acc))
points.sort()
print("parsed %d epochs (%d validation k); final greedy acc = %.2f%%"
      % (len(points), n, points[-1][1]))

# ---- plot (matplotlib bundled with Sage; PDF-safe) ----
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

xs = [int(p[0]) for p in points]
ys = [float(p[1]) for p in points]

fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.plot(xs, ys, color='purple', lw=float(1.5), marker='o', ms=float(3),
        label='greedy exact-match')
ax.set_xlabel('epoch')
ax.set_ylabel('validation greedy accuracy (%)')
ax.set_title(r'$p = 3$: validation greedy accuracy vs. epoch')
ax.grid(True, color='0.85')                 # solid gray, no alpha -> PDF-safe
ax.legend(loc='lower right', framealpha=1.0)
fig.tight_layout()
#fig.savefig('greedy_accuracy_p3_vs_epoch.png', dpi=int(150))
#fig.savefig('greedy_accuracy_p3_vs_epoch.pdf')
#print("saved greedy_accuracy_p3_vs_epoch.png / .pdf")
plt.show()

