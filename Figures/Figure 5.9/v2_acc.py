#!/usr/bin/env python
# coding: utf-8

# In[1]:


# plot_greedy_accuracy.sage
# Graph of validation greedy exact-match accuracy vs. epoch.
# Parses the per-epoch dumps eval.valid.elliptic_curve.<epoch> produced by training.
#
# Run:  sage plot_greedy_accuracy.sage
# Output: greedy_accuracy_vs_epoch.png   (+ returns the (epoch, acc) list)

import os, re, glob

# ---- point this at the folder holding eval.valid.elliptic_curve.* ----
DATA_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'

# Each block looks like:
#   problem={'k': 84538, ...}
#   answer=[-10, 1]
#   0 greedy
#     decoded=[-7, 1]
# We take exact-match of the FULL decoded list vs answer (the greedy_acc metric).
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

def epoch_of(path):
    return int(path.rsplit('.', 1)[1])

files = sorted([f for f in glob.glob(os.path.join(DATA_DIR, '../MahlerExperiments/v2/v2_fD/v2_fD run 3 (best run)/eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()],
               key=epoch_of)
if not files:
    raise FileNotFoundError('no eval.valid.elliptic_curve.* files found -- '
        'expected the MahlerExperiments folder inside the Figures folder '
        '(see the path at the top of this script)')

points = []
for f in files:
    acc, n = greedy_accuracy(f)
    if acc is not None:
        points.append((epoch_of(f), acc))

points.sort()
print("parsed %d epochs; final greedy acc = %.2f%%" % (len(points), points[-1][1]))

# ---- plot with matplotlib (bundled with Sage) for clean, centered axis labels ----
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

# Cast to plain Python types: under the Sage preparser numeric literals become
# Sage types (RealLiteral/Integer), which the matplotlib PDF backend can't serialize.#
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

xs = [int(p[0]) for p in points]
ys = [float(p[1]) for p in points]

fig, ax = plt.subplots(figsize=(9.0, 5.0))
ax.plot(xs, ys, color='blue', lw=float(1.5), marker='o', ms=float(3),
        label='greedy exact-match')
ax.set_xlabel('epoch')
ax.set_ylabel('validation greedy accuracy (%)')
ax.set_title('Validation greedy accuracy vs. epoch')
ax.grid(True, color='0.85')                    # solid gray, no alpha -> PDF-safe
ax.legend(loc='lower right', framealpha=1.0)
fig.tight_layout()
#fig.savefig('greedy_accuracy_vs_epoch.png', dpi=int(150))
#fig.savefig('greedy_accuracy_vs_epoch.pdf')
#print("saved greedy_accuracy_vs_epoch.png / .pdf")
plt.show()

# points is available for further use (e.g. filter to odd k, overlay a baseline line, ...)


# In[ ]:




