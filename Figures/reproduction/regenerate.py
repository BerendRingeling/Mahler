"""Rebuild the 14 revised Section 5 PDFs from the public run archive."""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_STORED, ZipFile

HERE = Path(__file__).resolve().parent
CONFIGS = {
    'rk': ('rk', 'rk_w', 'rk_wN', 'rk_wfN', 'rk_wfD'),
    'v2': ('v2_fD',),
    'v3': ('v3_fD',),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiments', type=Path, nargs='?', default=HERE.parent/'MahlerExperiments')
    parser.add_argument('--output-dir', type=Path, default=Path('regenerated-figures'))
    args = parser.parse_args()
    experiments = args.experiments.resolve()
    destination = args.output_dir.resolve()

    with TemporaryDirectory(prefix='mahler-figures-') as temp:
        work = Path(temp)
        for folder in ('inputs', 'outputs', 'work'):
            (work/folder).mkdir()
        for name in ('figure_styles_20.pdf', 'figure_styles_25.pdf'):
            shutil.copy2(HERE/name, work/'inputs'/name)
        shutil.copy2(HERE/'v2_validation_predictors.npz', work/'inputs/dataset_derived.npz')
        shutil.copy2(HERE/'hist.py', work/'work/hist.py')
        shutil.copy2(HERE/'mahler_v3_figures.py', work/'outputs/mahler_v3_figures.py')

        with ZipFile(work/'inputs/mahler_uploads.zip', 'w', ZIP_STORED) as archive:
            for family, configs in CONFIGS.items():
                for config in configs:
                    folder = experiments/family/config
                    logs = sorted(folder.glob('*/train.log'))
                    if len(logs) != 3:
                        parser.error(f'Expected three runs under {folder}')
                    for log in logs:
                        for file in (log, *(log.parent/f'eval.{split}.epochs_0_499.json.bz2'
                                            for split in ('valid', 'test1'))):
                            archive.write(file, 'mahler_uploads/' + file.relative_to(experiments).as_posix())

        def run(script, *arguments):
            subprocess.run([sys.executable, str(HERE/script), *arguments], cwd=work, check=True)

        for version in (3, 2):
            run(f'mahler_v{version}_figures.py', 'inputs/mahler_uploads.zip',
                'inputs/figure_styles_20.pdf', '--output-dir', 'outputs')
        run('recalc.py')
        run('finish_rk_plots.py')
        run('pearson25.py')
        run('plot_november.py')
        figures = sorted((work/'outputs').glob('*.pdf'))
        if len(figures) != 14:
            raise RuntimeError(f'Expected 14 generated PDFs, found {len(figures)}')
        destination.mkdir(parents=True, exist_ok=True)
        for figure in figures:
            shutil.copy2(figure, destination/figure.name)
        print(f'Wrote {len(figures)} PDFs to {destination}')

if __name__ == '__main__':
    main()
