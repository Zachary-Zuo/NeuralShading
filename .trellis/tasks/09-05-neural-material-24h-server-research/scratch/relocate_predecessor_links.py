"""一次性归档链接迁移；默认只审计，--apply 只在首次迁移时使用。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[4]
OLD = ROOT / '.trellis/tasks/09-05-neural-material-spatial-optimization-research'
ARCHIVE = ROOT / '.trellis/tasks/archive/2026-09/09-05-neural-material-spatial-optimization-research'
CURRENT = ROOT / '.trellis/tasks/09-05-neural-material-24h-server-research'
LINK = re.compile(r'(\[[^\]\n]*\]\()([^\)\n]+)(\))')


def local_target(raw: str) -> tuple[str, str] | None:
    if re.match(r'^(?:[a-zA-Z][\w+.-]*:|#|/)', raw):
        return None
    match = re.fullmatch(r'(.*?)(:\d+)?(#[^\s]*)?', raw)
    assert match is not None
    return match[1], (match[2] or '') + (match[3] or '')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    assert ARCHIVE.is_dir() and not OLD.exists()
    if args.apply:
        for destination in ARCHIVE.rglob('*.md'):
            original = OLD / destination.relative_to(ARCHIVE)

            def replace(match: re.Match[str]) -> str:
                parsed = local_target(match[2])
                if parsed is None:
                    return match[0]
                target, suffix = parsed
                resolved = (original.parent / target).resolve()
                if resolved.is_relative_to(OLD):
                    resolved = ARCHIVE / resolved.relative_to(OLD)
                relative = Path(os.path.relpath(resolved, destination.parent)).as_posix()
                return match[1] + relative + suffix + match[3]

            text = destination.read_text(encoding='utf-8')
            updated = LINK.sub(replace, text)
            if updated != text:
                destination.write_text(updated, encoding='utf-8', newline='\n')
        metadata = ARCHIVE / 'task.json'
        data = json.loads(metadata.read_text(encoding='utf-8-sig'))
        data['relatedFiles'] = [p.replace(OLD.relative_to(ROOT).as_posix(), ARCHIVE.relative_to(ROOT).as_posix()) for p in data['relatedFiles']]
        data['commit'] = 'e044156d9031f186551e05ff19436fbabfbd2b59'
        metadata.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    missing = []
    count = 0
    for directory in (ARCHIVE, CURRENT):
        for file in directory.rglob('*.md'):
            for match in LINK.finditer(file.read_text(encoding='utf-8')):
                parsed = local_target(match[2])
                if parsed is None:
                    continue
                count += 1
                if not (file.parent / parsed[0]).exists():
                    missing.append({'file': str(file.relative_to(ROOT)), 'target': match[2]})
    print(json.dumps({'local_links_checked': count, 'missing': missing}, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
