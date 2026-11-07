"""Rebuild the five corrected Mahler p=3 figures from the raw run archive.

Run with SageMath's Python, with NumPy and pypdf installed:
    sage -python mahler_v3_figures.py mahler_uploads.zip "Mahler (20).pdf" --output-dir figures
The manuscript PDF supplies the original embedded vector plot styles.
"""
import argparse
import bz2
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ContentStream, DecodedStreamObject, DictionaryObject, FloatObject, NameObject,
)


def load_run(archive, run, split):
    directory = f"mahler_uploads/v3/v3_fD/{run}/"
    with archive.open(directory + f"eval.{split}.epochs_0_499.json.bz2") as member:
        with bz2.open(member, "rt", encoding="utf-8") as stream:
            rows = json.load(stream).values()

    k = []
    targets = []
    correct = []
    for row in rows:
        k.append(row["problem"]["k"])
        a, b = row["answer"]
        targets.append((a, b))
        match = np.zeros(500, dtype=bool)
        for epoch, guess in row["guesses_greedy"].items():
            if (
                isinstance(guess, list)
                and len(guess) == 2
                and all(isinstance(value, int) for value in guess)
                and guess[1]
            ):
                match[int(epoch)] = guess[0] * b == a * guess[1]
        correct.append(match)
    return {"k": np.array(k), "y": np.array(targets), "exact": np.array(correct)}


def fs(k):
 d={}; q=2
 while q*q<=k:
  while k%q==0: d[q]=d.get(q,0)+1;k//=q
  q+=1
 if k>1:d[k]=1
 return d
def sets(k):
 x=fs(k); d=Counter(fs(abs(k-4)));d.update(fs(k+4))
 a2=all(e<2 for e in d.values()); b=all(e<2 for p,e in d.items() if p!=2)
 c=all(e<2 for p,e in d.items() if p%4==3)
 bp=b and k%8!=4;cp=c and k%8!=4
 cpp=cp and all(e<3 or p%3==2 for p,e in (Counter(x)+d).items())
 return [k%3!=0 and all(e<2 for e in (Counter(x)+d).values()),k%3!=0 and a2,a2,b,bp,bp and all(e<3 or p%3==2 for p,e in x.items()),c,cp,cpp]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("manuscript", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    runs=['1000260_9','1000260_10','1000260_11']
    with ZipFile(args.archive) as archive:
        for run in runs:
            log = archive.read(f"mahler_uploads/v3/v3_fD/{run}/train.log").decode()
            assert "Creating valid iterator" in log and "Creating test1 iterator" in log
            assert "tokenized_validate_base_3.data" in log
            assert "tokenized_test_base_3.data" in log
        v = [load_run(archive, run, "valid") for run in runs]
        t = [load_run(archive, run, "test1") for run in runs]
    epochs=[int(a['exact'][:,:300].sum(0).argmax()) for a in v]
    assert epochs==[244,216,298]
    def labels(a):
        assert (a['y'][:,1]==1).all()
        return -a['y'][:,0]
    for group in (v,t):
        for a in group:
            assert np.array_equal(a['k'],group[0]['k'])
            assert np.array_equal(labels(a),labels(group[0]))
    assert not set(v[0]['k']) & set(t[0]['k'])
    assert np.allclose([a['exact'][:,e].mean()*100 for a,e in zip(t,epochs)],[49.2,48.4,48.8])
    yv,yt=map(labels,(v[0],t[0]))
    mv,mt=[np.array([sets(int(k)) for k in group[0]['k']]) for group in (v,t)]
    lv=np.isin(yv,[0,1]); lt=np.isin(yt,[0,1])
    def outside(a,m,y):
        return ~m[:,-1] & ~np.isin(a['k']%27,[0,5,22]) & np.isin(y,[0,1])
    ov=outside(v[0],mv,yv)
    ot=outside(t[0],mt,yt)
    assert (sum(ov),sum(ot))==(540,551)
    def curve(mask):
        return 100*np.array([a['exact'][mask].mean(0) for a in v])
    manuscript = PdfReader(args.manuscript)
    conditional = lambda mask: [curve(mask & group) for group in (yv == 1, yv == 0, yv > 1)]
    all_examples = np.ones(len(yv), dtype=bool)

    update_plot(
        manuscript, out, "v3_fd_validation_accuracy", 26, 11, 10, 5,
        [curve(all_examples)], [1730], [2224],
        baseline=(3234, 100 * (yv == 1).mean()), markers=(2225, 3234),
    )
    update_plot(
        manuscript, out, "v3_fd_conditional_accuracy", 27, 13, 0, 20,
        conditional(all_examples), [1678, 2692, 3706], [4198, 4693, 5182],
        clamp_bands=True,
    )
    update_plot(
        manuscript, out, "v3_fd_set_a_accuracy", 28, 15, 0, 20,
        conditional(mv[:, 0]), [1678, 2692, 3706], [4179, 4670, 5151],
        clamp_bands=True,
    )
    update_plot(
        manuscript, out, "v3_fd_sets_comparison", 29, 17, 40, 10,
        [curve(lv & mv[:, j]) for j in range(9)], [],
        [1222, 1721, 2221, 2726, 3227, 3724, 4229, 4732, 5230],
    )
    update_plot(
        manuscript, out, "v3_fd_complement_accuracy", 30, 19, 40, 10,
        [curve(ov)], [1726], [2256], baseline=(1730, 100 * (yv[ov] == 1).mean()),
    )


def update_plot(
    manuscript, out, filename, page, form_number, ymin_tick, tick_step,
    curves, band_ends, line_ends, *, baseline=None, markers=None, clamp_bands=False,
    tick_indices=(32, 66, 414, 448), x_tick_step=50, min_path_points=400,
):
    """Replace only the numerical paths in a known form in Mahler (20)."""
    form = manuscript.pages[page - 1]["/Resources"]["/XObject"][f"/Im{form_number}"]
    content = ContentStream(form, manuscript)
    ops = content.operations
    x_first, x_next, y_first, y_next = tick_indices
    assert all(ops[i][1] == b"m" for i in tick_indices)
    x0 = float(ops[x_first][0][0])
    dx = (float(ops[x_next][0][0]) - x0) / x_tick_step
    y0 = float(ops[y_first][0][1])
    dy = (float(ops[y_next][0][1]) - y0) / tick_step
    xs = x0 + dx * np.arange(curves[0].shape[1])

    def points(values):
        return list(zip(xs, y0 + dy * (values - ymin_tick)))

    def path(coordinates):
        return [
            ([FloatObject(x), FloatObject(y)], b"m" if i == 0 else b"l")
            for i, (x, y) in enumerate(coordinates)
        ]

    replacements = []
    def replace_path(end, coordinates, fill=False):
        assert ops[end][1] == (b"f" if fill else b"S")
        start = end
        while start and ops[start - 1][1] in (b"m", b"l", b"h"):
            start -= 1
        assert ops[start][1] == b"m" and end - start > min_path_points
        replacements.append((start, end, path(coordinates) + ([([], b"h")] if fill else [])))

    for data, end in zip(curves, line_ends):
        replace_path(end, points(data.mean(axis=0)))
    for data, end in zip(curves, band_ends):
        mean = data.mean(axis=0)
        spread = data.std(axis=0, ddof=1)
        low, high = mean - spread, mean + spread
        if clamp_bands:
            low, high = np.clip(low, 0, 100), np.clip(high, 0, 100)
        replace_path(end, points(low) + points(high)[::-1], fill=True)
    if baseline:
        clip, value = baseline
        line = clip + 8
        assert [op for _, op in ops[line:line + 3]] == [b"m", b"l", b"S"]
        height = FloatObject(y0 + dy * (value - ymin_tick))
        for i in (line, line + 1):
            xy, op = ops[i]
            replacements.append((i, i + 1, [([xy[0], height], op)]))
    if markers:
        start, stop = markers
        coordinates = points(curves[0].mean(axis=0))
        matrix_indices = [i for i in range(start, stop) if ops[i][1] == b"cm"]
        assert len(matrix_indices) == len(coordinates)
        previous_x, previous_y = 0, 0
        for i, (x, y) in zip(matrix_indices, coordinates):
            args, op = ops[i]
            updated = args[:4] + [FloatObject(x - previous_x), FloatObject(y - previous_y)]
            replacements.append((i, i + 1, [(updated, op)]))
            previous_x, previous_y = x, y
    for start, end, replacement in sorted(replacements, reverse=True):
        ops[start:end] = replacement
    content.operations = ops

    write_form(form, content, out / f"{filename}_corrected.pdf")


def write_form(form, content, filename):
    writer = PdfWriter()
    _, _, width, height = form["/BBox"]
    sheet = writer.add_blank_page(float(width), float(height))
    plot = DecodedStreamObject()
    for key, value in form.items():
        if key not in ("/Filter", "/Length", "/DecodeParms"):
            plot[key] = value.clone(writer)
    plot.set_data(content.get_data())
    sheet[NameObject("/Resources")] = DictionaryObject({
        NameObject("/XObject"): DictionaryObject({NameObject("/Plot"): writer._add_object(plot)})
    })
    drawing = DecodedStreamObject()
    drawing.set_data(b"/Plot Do")
    sheet[NameObject("/Contents")] = writer._add_object(drawing)
    writer.write(filename)


if __name__ == "__main__":
    main()
