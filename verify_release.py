# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""Verify saved release files and the manuscript figure index, without training."""
from pathlib import Path
import ast
import hashlib
import json
import tomllib

ROOT = Path(__file__).resolve().parent


def main():
    release = json.loads((ROOT / 'release_manifest.json').read_text())
    files = release['files']
    for rel, expected in files.items():
        path = ROOT / rel
        if not path.is_file():
            raise ValueError(f'Missing release file: {rel}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Checksum mismatch: {rel}')

    manuscript = json.loads((ROOT / 'figure_manifest.json').read_text())
    assert manuscript['manuscript_revision'] == release['manuscript_revision']
    figures = manuscript['figures']
    assert [r['number'] for r in figures] == list(range(1, 30))
    assert len({r['label'] for r in figures}) == 29
    assert {p.name for p in (ROOT / 'reference_figures').glob('*.pdf')} == {r['file'] for r in figures}
    for row in figures:
        path = ROOT / 'reference_figures' / row['file']
        assert path.read_bytes().startswith(b'%PDF-'), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256'], path
        assert (ROOT / row['directory'] / 'README.md').is_file(), row['directory']

    count = 0
    for rel in files:
        path = ROOT / rel
        if path.suffix != '.py':
            continue
        source = path.read_text()
        ast.parse(source, filename=rel)
        count += 1
        if '# /// script' in source:
            lines = source.split('# /// script', 1)[1].split('# ///', 1)[0].strip().splitlines()
            metadata = tomllib.loads('\n'.join(line.removeprefix('#').removeprefix(' ') for line in lines))
            assert 'dependencies' in metadata, rel
        elif not any((parent / 'pyproject.toml').exists() for parent in path.parents if parent.is_relative_to(ROOT)):
            raise ValueError(f'Standalone script lacks PEP 723 metadata: {rel}')
    print(f'PASS: {len(files)} file hashes, {count} Python sources, 29 manuscript figures.')


if __name__ == '__main__':
    main()
