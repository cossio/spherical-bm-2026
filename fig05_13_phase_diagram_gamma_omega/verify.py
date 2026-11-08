# /// script
# requires-python = ">=3.14"
# dependencies = ["numpy==2.4.4", "pymupdf>=1.26"]
# ///
"""Verify the two distinct KL tables and pixel-identical manuscript figures."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf

ROOT = Path(__file__).resolve().parent
CASES = {
    "reverse": ("pp_rev_kl_2gamma.csv", "phase_diagram_gamma_omega.pdf", (400,4), "9a0a6c63cdf5cce6c000b17333bd5d847b59155ace3e7dc8b4070e1424cf068b"),
    "forward": ("pp_fwd_kl_4gamma.csv", "fig_fwd_pp_kl_phase_diagram.pdf", (600,5), "a5d6568cca4d531f3c443024947543aedde08bb304268cbd0b2bc7dce314538a"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "reproduced")
    parser.add_argument("--reverse-curves", type=Path, help="Optional independently regenerated reverse CSV")
    parser.add_argument("--forward-curves", type=Path, help="Optional independently regenerated forward CSV")
    args = parser.parse_args()
    records = {}
    for mode,(table,pdf,shape,reference_hash) in CASES.items():
        saved = ROOT / "data" / mode / table
        values = np.loadtxt(saved, delimiter=",", skiprows=1)
        assert values.shape == shape and np.isfinite(values).all()
        assert np.all(np.diff(values[:,0]) > 0) and np.all(values[:,1:] >= -1e-12)
        regenerated = getattr(args, f"{mode}_curves")
        if regenerated is not None:
            np.testing.assert_allclose(np.loadtxt(regenerated,delimiter=",",skiprows=1),values,rtol=2e-12,atol=2e-12)
        reference = ROOT.parent / "reference_figures" / pdf
        assert hashlib.sha256(reference.read_bytes()).hexdigest() == reference_hash
        with pymupdf.open(reference) as a, pymupdf.open(args.figure_dir/pdf) as b:
            assert len(a) == len(b) == 1
            ap,bp = a[0].get_pixmap(dpi=144),b[0].get_pixmap(dpi=144)
            assert (ap.width,ap.height,ap.n) == (bp.width,bp.height,bp.n)
            assert ap.samples == bp.samples, f"Rendered {mode} figure differs from manuscript reference"
            assert not a[0].get_images() and not b[0].get_images()
        records[mode] = {"table_shape":shape,"reference_sha256":reference_hash,
            "rendered_pixels_identical_at_144dpi":True,"curves_recomputed":regenerated is not None}
    print(json.dumps({"status":"PASS","checks":records},indent=2))


if __name__ == "__main__":
    main()
