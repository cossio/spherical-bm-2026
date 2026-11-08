# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4"]
# ///
"""Audit the exported saved data without rerunning the production simulation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    hashes = json.loads((ROOT/'checksums.json').read_text())['files']
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    for path in (ROOT/'results').glob('case_*.npz'):
        with np.load(path, allow_pickle=False) as data:
            assert json.loads(str(data['metadata'])) == json.loads(path.with_suffix('.json').read_text()), path.name
    for script in ['scripts/analyze.py', 'validation/independent_results_audit.py']:
        subprocess.run(['uv','run',str(ROOT/script),'--require-complete'], check=True, capture_output=True, text=True)
    original = json.loads((ROOT/'validation/summary.json').read_text())
    recomputed = json.loads((ROOT/'generated/summary.json').read_text())
    assert original == recomputed, 'Recomputed numerical summary differs'
    audit = json.loads((ROOT/'generated/independent_results_report.json').read_text())
    assert audit['complete'] and audit['run_count'] == 24
    assert audit['analyzer_summary_fields_compared'] == 288
    refinement_checks = 0
    for coarse_tag, fine_tag, filename in [('0p05','0p025','refinement.csv'),
                                          ('0p025','0p0125','refinement_dt0p025_to0p0125.csv')]:
        with (ROOT/'theory'/filename).open() as stream:
            recorded = list(csv.DictReader(stream))
        for row in recorded:
            coarse = np.genfromtxt(ROOT/'theory'/f"{row['case']}_dt{coarse_tag}_optimized.csv",names=True,delimiter=',')
            fine = np.genfromtxt(ROOT/'theory'/f"{row['case']}_dt{fine_tag}_optimized.csv",names=True,delimiter=',')[::2]
            assert np.array_equal(coarse['t'],fine['t'])
            key=row['observable']
            difference=coarse[key]-fine[key]
            actual={'max_abs_difference':np.max(np.abs(difference)),
                    'rms_difference':np.sqrt(np.mean(difference*difference)),
                    'coarse_final':coarse[key][-1],'fine_final':fine[key][-1]}
            for field,value in actual.items():
                assert abs(value-float(row[field])) < 1e-13, (filename,key,field)
                refinement_checks += 1
    print(json.dumps({'passed':True,'file_hashes':len(hashes),'production_settings':24,
                      'independent_summary_fields':288,'theory_refinement_values':refinement_checks,
                      'production_source_sha256':hashlib.sha256((ROOT/'scripts/run_sgd.py').read_bytes()).hexdigest()},indent=2))


if __name__ == '__main__':
    main()
