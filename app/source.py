# SPDX-License-Identifier: AGPL-3.0-only
"""Offer application source without packaging local runtime data or models."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

ROOT_FILES = ('LICENSE', 'NOTICE', 'README.md', 'READMEJ.md', 'HANDOFF.md',
              '.gitignore', 'camera.yaml', 'requirements.txt', 'requirements-quant.txt',
              'create_models.sh', 'start_all.sh', 'stop_all.sh')
SOURCE_TYPES = {'.py', '.sh', '.html', '.css', '.js', '.txt', '.cpp'}


def source_archive(root):
    root = Path(root).resolve()
    files = [root / name for name in ROOT_FILES]
    for directory in ('app', 'scripts', 'tests', 'licenses'):
        files.extend(p for p in (root / directory).rglob('*')
                     if p.suffix in SOURCE_TYPES and '__pycache__' not in p.parts)
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            # Only regular files within this source tree; do not follow symlinks.
            if not path.is_file() or path.is_symlink():
                continue
            if any(parent.is_symlink() for parent in path.parents if parent != root):
                continue
            if not path.resolve().is_relative_to(root):
                continue
            archive.write(path, str(Path('yolodemo') / path.relative_to(root)))
    return output.getvalue()
