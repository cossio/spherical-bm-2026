# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Check the optimized DMFT loop against its prior loop and saved curve prefixes."""
import argparse
import csv
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open() as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--julia', default='julia')
    args = parser.parse_args()
    output = ROOT / 'generated' / 'theory_check'
    output.mkdir(parents=True, exist_ok=True)
    errors = {}
    for case, c, s0, gamma in [('A','1.5,.5','.30,.25','.5'), ('B','1.8,.2','.30,.15','.5'),
                              ('C','1','.30','.5'), ('D','1','.10','1.5')]:
        results = []
        for name in ['msr_reference', 'original_loop_check']:
            target = output / f'{case}_{name}.csv'
            command = [args.julia, '--startup-file=no', str(ROOT/'theory'/f'{name}.jl'),
                       '--eta=Inf', '--beta=1', '--nu=.3', f'--gamma={gamma}',
                       f'--c={c}', f'--s0={s0}', '--dt=.05', '--Tmax=2', f'--outfile={target}']
            subprocess.run(command, check=True, capture_output=True, text=True)
            results.append(rows(target))
        optimized, original = results
        assert len(optimized) == len(original) == 41
        error = max(abs(a[key]-b[key]) for a,b in zip(optimized,original) for key in a)
        assert error < 1e-13, (case, error)
        full = rows(ROOT/'theory'/f'{case}_dt0p05_optimized.csv')
        # The shorter run repeats its penultimate kappa at its last time point.
        prefix_error = max(abs(a[key]-b[key]) for a,b in zip(optimized[:-1],full) for key in a)
        assert prefix_error < 1e-13, (case, prefix_error)
        errors[case] = {'original_loop_max_error':error, 'saved_prefix_max_error':prefix_error}
    report = {'passed':True, 'julia_version':subprocess.check_output([args.julia,'--version'],text=True).strip(),
              'cases':errors, 'comparison_horizon':2, 'step':.05,
              'endpoint':'Final shortened-run kappa excluded only from full-run prefix comparison.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
