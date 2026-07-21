#!/usr/bin/env python
# coding: utf-8

# In[1]:


# plot_L2_snapshot.sage
#
# Experiment 67 (PB1: input k, w, factor(Delta); the model learns r_k).
# Single-frame snapshot of the "movie": a histogram of the model's guessed
# value of  L(E,2) = 4*pi^2* |n_k| * mu_k / N_k  at ONE fixed epoch, drawn on
# top of the true L(E,2) distribution (grey) as a reference.
#
#   * the model outputs a guess for  |n_k| = 1/|r_k|  (the large integer);
#   * mu_k is the Mahler measure of  x+1/x+y+1/y+k  (hypergeometric series);
#   * N_k is the conductor of E_k, read from output_full.txt (field [3][0]);
#   * L_guess = 4 pi^2 * mu_k / N_k * (model's guessed |n_k|).
#
# By the functional equation, L(E,2) = 4 pi^2 L'(E,0)/(w N) and mu_k = r_k L'(E,0),
# so this formula is exactly L(E,2): the true values form a tidy O(1) bump.
#
# Run:  sage plot_L2_snapshot.sage
import os, re, ast, glob, math
import numpy as np

# ================= configure =================
EPOCH        = 260                       # <-- fixed epoch to snapshot
EXP_DIR      = '../MahlerExperiments/rk/rk_wfD/size/rk_wfD (size)'    # folder holding eval.valid.elliptic_curve.*
OUTPUT_FULL  = '../MahlerExperiments/data/output_full.txt'         # file with the conductor N_k
LO, HI, NB   = 0.40, 1.80, 70            # histogram range and bin count
# =============================================

HERE = os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.'
def find(name):
    for c in [name, os.path.join(HERE, name)]:
        if os.path.exists(c):
            return c
    hits = glob.glob(os.path.join(HERE, '**', os.path.basename(name)), recursive=True)
    if hits:
        return hits[0]
    raise FileNotFoundError(name)

# locate the eval file for the chosen epoch
eval_dir = None
for c in [os.path.join(HERE, EXP_DIR), HERE]:
    if [f for f in glob.glob(os.path.join(c, 'eval.valid.elliptic_curve.*')) if f.rsplit('.', 1)[-1].isdigit()]:
        eval_dir = c; break
if eval_dir is None:
    hit = [f for f in glob.glob(os.path.join(HERE, '**', 'eval.valid.elliptic_curve.*'), recursive=True) if f.rsplit('.', 1)[-1].isdigit()]
    eval_dir = os.path.dirname(hit[0])
eval_file = os.path.join(eval_dir, 'eval.valid.elliptic_curve.%d' % EPOCH)
out_full  = find(OUTPUT_FULL)

# ---- parse the eval file: k, true |n_k|, greedy-predicted |n_k| ----
REC = re.compile(
    r"'k':\s*(?P<k>\d+).*?answer=\[(?P<sa>-?\d+),\s*(?P<na>\d+)\]"
    r".*?greedy\s*\n\s*decoded=\[(?P<sg>-?\d+),\s*(?P<ng>\d+)\]", re.S)
true_n, pred_n = {}, {}
for m in REC.finditer(open(eval_file).read()):
    k = int(m.group('k'))
    true_n[k] = int(m.group('na'))
    pred_n[k] = int(m.group('ng'))
valk = set(true_n)

# ---- conductor N_k from output_full.txt (field [3][0]) ----
frac = re.compile(r'Fraction\((-?\d+),\s*(-?\d+)\)')
N = {}
with open(out_full) as fh:
    for line in fh:
        i = line.find(',')
        if i < 1:
            continue
        try:
            k = int(line[1:i])
        except ValueError:
            continue
        if k not in valk:
            continue
        row = ast.literal_eval(frac.sub(r'(\1,\2)', line.strip()))
        N[k] = int(row[3][0])

# ---- Mahler measure mu_k (hypergeometric series) ----
def mu(k):
    k = abs(float(k)); z = 16.0 / k / k; term = 1.0; F = 1.0; n = 0
    while abs(term) > 1e-17 and n < 300:
        term *= (1.5 + n)**2 * (n + 1) * z / ((2.0 + n)**3); F += term; n += 1
    return math.log(k) - (2.0 / (k * k)) * F

ks = sorted(k for k in valk if k in N)
C  = {k: 4.0 * math.pi**2 * mu(k) / N[k] for k in ks}     # L = C_k * |n_k|
Ltrue  = np.array([C[k] * true_n[k] for k in ks], dtype=float)
Lguess = np.array([C[k] * pred_n[k] for k in ks], dtype=float)
Lguess[~np.isfinite(Lguess)] = 1e9                        # push overflow out of range

# factor-2 "size accuracy" at this epoch (same criterion used throughout)
with np.errstate(divide='ignore', invalid='ignore'):
    lr = np.log(np.where(Lguess > 0, Lguess, np.nan)) - np.log(Ltrue)
size_acc = float(100.0 * np.nanmean(np.abs(lr) < math.log(2.0)))

# ---- histograms ----
edges = np.linspace(LO, HI, NB + 1)
cen   = 0.5 * (edges[:-1] + edges[1:]); w = float(edges[1] - edges[0])
tc, _ = np.histogram(Ltrue,  bins=edges)
gc, _ = np.histogram(Lguess, bins=edges)

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
TRUE, GUESS = '#9aa0a6', '#1f77b4'
fig, ax = plt.subplots(figsize=(9.2, 5.3))
ax.fill_between(cen, tc, step='mid', color=TRUE, alpha=float(0.55), zorder=int(1),
                label=r'true $L(E_k,2)$ (data)')
ax.bar(cen, gc, width=float(w * 0.96), color=GUESS, alpha=float(0.85), zorder=int(3),
       edgecolor='white', linewidth=float(0.3),
       label=r"model's predicted $L(E_k,2)$")
ax.set_xlim(float(LO), float(HI))
ax.set_ylim(int(0), float(max(tc.max(), gc.max()) * 1.18))
ax.set_xlabel(r'$L(E_k,2)=4\pi^{2}\,|n_k|\,\mu_k/N_k$')
ax.set_ylabel(r'count  (over %d validation $k$)' % len(ks))
ax.grid(True, axis='y', color='0.92'); ax.set_axisbelow(True)
ax.legend(loc='upper right', framealpha=float(0.9))
ax.set_title(r'Model guesses of $L(E_k,2)$ from predicted $|n_k|$  ($\mathrm{epoch}\ %d$)' % EPOCH)
#ax.text(0.015, 0.955,
#        'epoch %3d\nsize acc  %5.1f%%\nmedian    %.3f\n(true med %.3f)'
#        % (EPOCH, size_acc, float(np.median(np.clip(Lguess, 0, HI))), float(np.median(Ltrue))),
#        transform=ax.transAxes, va='top', ha='left', fontsize=int(10), family='monospace',
#        bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=float(0.9)))
fig.tight_layout()
#fig.savefig('L2_snapshot_ep%d.png' % EPOCH, dpi=int(140))
#fig.savefig('L2_snapshot_ep%d.pdf' % EPOCH)
#print('saved L2_snapshot_ep%d.png / .pdf   (size acc %.1f%%, %d curves)'
#      % (EPOCH, size_acc, len(ks)))
plt.show()


# In[ ]:




