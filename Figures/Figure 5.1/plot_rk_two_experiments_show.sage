# plot_rk_two_experiments_show.sage
#
# Compare two experiments that learn r_k (answer = [sign, |n_k|]) as a function
# of epoch: one with input k only, one with input k and the root number w.
#
# "Accuracy" here is the size criterion for learning r_k:  a model output s_k
# is correct if it has the same sign as r_k and  |r_k|/2 < |s_k| < 2|r_k|.
# The eval stores the magnitude |n_k| = 1/|r_k| (the large integer), and the
# factor-2 window is the same on |n_k| as on |r_k| (since s <-> 1/s), so with
# ACCEPT = 2 the test below,  |log(guess) - log(true)| < log(ACCEPT), is exactly
# this criterion.  Sign-only accuracy is also plotted for reference.
#
# This version only shows the figure in a window; nothing is saved to disk.
#
# ---> set the two folder paths below <---
#
# Run:  sage plot_rk_two_experiments_show.sage
import os, re, glob, math
import numpy as np
# ============ configure ============
EXP1_DIR   = '../MahlerExperiments/rk/rk/size/rk run 2 (size) (best run)'          # folder with eval.valid.* files, input = k
EXP2_DIR   = '../MahlerExperiments/rk/rk_w/size/rk_w (size)'         # folder with eval.valid.* files, input = k, w
EXP1_LABEL = r'input $k$'
EXP2_LABEL = r'input $k,\ \omega_k$'
ACCEPT     = 2.0                    # magnitude tolerance factor
# ===================================
def epoch_of(p): return int(p.rsplit('.', 1)[1])
def eval_files(d):
    return sorted([f for f in glob.glob(os.path.join(d, '**', 'eval.valid.elliptic_curve.*'), recursive=True) if f.rsplit('.', 1)[-1].isdigit()],
                  key=epoch_of)
# answer=[a0, a1]  ... greedy \n decoded=[g0, g1]
BLOCK = re.compile(
    r"answer=\[(?P<a0>-?\d+),\s*(?P<a1>-?\d+)\].*?greedy\s*\n\s*decoded=\[(?P<g0>-?\d+),\s*(?P<g1>-?\d+)\]",
    re.S)
logA = math.log(ACCEPT)
def curve(d):
    files = eval_files(d)
    E = []; size_acc = []; sign_acc = []
    for f in files:
        n = size_ok = sign_ok = 0
        for m in BLOCK.finditer(open(f).read()):
            a0 = int(m.group('a0')); a1 = int(m.group('a1'))
            g0 = int(m.group('g0')); g1 = int(m.group('g1'))
            n += 1
            sgn = (a0 > 0) == (g0 > 0)
            sign_ok += int(sgn)
            if a1 != 0 and g1 != 0 and sgn and abs(math.log(abs(g1)) - math.log(abs(a1))) < logA:
                size_ok += 1
        E.append(epoch_of(f))
        size_acc.append(float(100.0 * size_ok / n) if n else float('nan'))
        sign_acc.append(float(100.0 * sign_ok / n) if n else float('nan'))
    o = sorted(range(len(E)), key=lambda i: E[i])
    return ([int(E[i]) for i in o], [size_acc[i] for i in o], [sign_acc[i] for i in o])
E1, sz1, sg1 = curve(EXP1_DIR)
E2, sz2, sg2 = curve(EXP2_DIR)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(9.2, 5.4))
ax.plot(E1, sz1, color='#1f77b4', lw=float(1.8), label=EXP1_LABEL + r'  (size acc.)')
ax.plot(E2, sz2, color='#d62728', lw=float(1.8), label=EXP2_LABEL + r'  (size acc.)')
ax.set_xlabel('epoch')
ax.set_ylabel(r'validation accuracy   (%)')
ax.set_title(r'Learning $r_k$: input $k$ vs. input $k,\omega_k$, per epoch')
ax.grid(True, color='0.92')
ax.set_ylim(int(0), int(102))
ax.legend(loc='lower right', fontsize=int(9))
fig.tight_layout()
plt.show()
