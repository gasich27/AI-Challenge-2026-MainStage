import ast
import json
import re
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebooks = list(ROOT.glob('archive/*.ipynb')) + list(ROOT.glob('labels/*/*.ipynb'))
    errors = []
    for path in notebooks:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != 'code':
                continue
            code = '\n'.join('' if line.lstrip().startswith(('%', '!')) else line
                             for line in cell.source.splitlines())
            try:
                ast.parse(code)
            except SyntaxError as error:
                errors.append(f'{path.relative_to(ROOT)} cell {i}: {error}')
    modules = [p for folder in ['src', 'scripts', 'tests', 'labels'] for p in (ROOT/folder).rglob('*.py')]
    for path in modules:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    for path in [ROOT/'README.md', *ROOT.glob('labels/*/README.md'), *ROOT.glob('docs/**/*.md')]:
        text = path.read_text(encoding='utf8')
        for target in re.findall(r'\]\(([^)]+)\)', text):
            target = target.strip('<>').split('#')[0]
            if not target or '://' in target:
                continue
            if not (path.parent / target).exists():
                errors.append(f'{path.relative_to(ROOT)}: нет {target}')
    print(json.dumps({'notebooks':len(notebooks), 'python_files':len(modules), 'errors':errors}, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
