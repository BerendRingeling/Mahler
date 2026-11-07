#!/usr/bin/env python3
"""Render the native table and plot as a combined PNG preview.

Optional: requires pdflatex with standalone/booktabs and pdftoppm.
The manuscript should use the original LaTeX fragment, not this PNG.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fragment", type=Path, default=Path("figure_5_7_replacement.tex"))
    parser.add_argument("--figure-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("v3_fd_sets_comparison_preview.png"))
    args = parser.parse_args()
    fragment = args.fragment.read_text()
    inner = fragment.split(r"\begin{figure}[t]", 1)[1].split(r"\caption", 1)[0]
    graphic_dir = args.figure_dir.resolve().as_posix() + "/"
    tex = (r"\documentclass[12pt,border=5pt]{standalone}" + "\n"
           r"\usepackage{amsmath,amssymb,graphicx,booktabs}" + "\n"
           r"\graphicspath{{" + graphic_dir + "}}\n"
           r"\begin{document}\begin{minipage}{17.7cm}" + "\n" + inner + "\n"
           r"\end{minipage}\end{document}" + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3_sets_preview_") as temporary:
        work = Path(temporary)
        (work / "preview.tex").write_text(tex)
        completed = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
                                   cwd=work, capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-6000:])
        log = (work / "preview.log").read_text(errors="replace")
        warnings = [line for line in log.splitlines() if any(x in line for x in ("LaTeX Warning", "Overfull", "Underfull"))]
        if warnings:
            raise RuntimeError("\n".join(warnings))
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "200", str(work / "preview.pdf"),
                        str(work / "preview")], check=True, capture_output=True)
        shutil.copyfile(work / "preview.png", args.output)
    print(json.dumps({"created":str(args.output),"latex_warnings":warnings}))


if __name__ == "__main__":
    main()
