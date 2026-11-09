"""Prune reproducible historical plots; preserve original states and key figures.

Default is a dry run. The allowlist is limited to the five historical groups,
never the current score-mask run. --apply writes an exact removal manifest and
checksums the estimation inputs/results before and after cleanup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent / 'outputs'
GROUPS = ('full_feedback_four_pairs_20261012', 'full_feedback_four_pairs_ransac8',
          'full_feedback_four_pairs_ransac8_depth_fit_weighted',
          'full_feedback_four_pairs_ransac8_depth_fit_masked_repair',
          'full_feedback_four_pairs_ransac8_depth_fit_global_mask_repair')
STAGE_DIRS = {'sparse_motion_visualization', 'ownership_stage_comparison',
              'feedback_match_visualization', 'feedback_final_visualization',
              'motion_visualization'}
PROTECTED = {'pairwise_state.npz', 'score_maps.npz', 'summary.json', 'cycles.json', 'config.yaml'}


def plan_cleanup():
    plan, protected = [], []
    for name in GROUPS:
        group = ROOT / name
        if not group.is_dir() or group.is_symlink():
            continue
        for path in sorted(group.rglob('*')):
            if not path.is_file() or path.is_symlink() or 'source_snapshot' in path.parts:
                continue
            if not path.resolve().is_relative_to(group.resolve()):
                raise ValueError(f'Path leaves its experiment group: {path}')
            if path.name in PROTECTED or path.suffix == '.npz':
                protected.append(path)
                continue
            reason = None
            if path.suffix == '.png':
                keep = (path.name == 'render_final.png' or
                        path.parent.name == 'comparisons' or
                        (path.parent.name in STAGE_DIRS
                         and path.name == 'overview.png'))
                if not keep:
                    reason = 'redundant stage/component/diagnostic image'
            if reason:
                stat = path.stat()
                plan.append(dict(path=str(path.relative_to(ROOT)), bytes=stat.st_size,
                                 mtime_ns=stat.st_mtime_ns, reason=reason))
    return plan, protected


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'cleanup_manifest.json')
    args = parser.parse_args()
    plan, protected = plan_cleanup()
    totals = dict(files=len(plan), png_files=sum(p['path'].endswith('.png') for p in plan),
                  bytes=sum(p['bytes'] for p in plan), protected_files=len(protected))
    print(json.dumps(totals), flush=True)
    if not args.apply:
        print('Dry run; no files changed.')
        return
    manifest = args.manifest.resolve()
    if not manifest.is_relative_to(ROOT.resolve()):
        raise ValueError('The cleanup manifest must be inside this experiment outputs directory')
    checksums = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    record = dict(totals=totals, removed=plan, protected_sha256=checksums, completed=False)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record, indent=2))
    for entry in plan:
        path = ROOT / entry['path']
        stat = path.stat()
        if stat.st_size != entry['bytes'] or stat.st_mtime_ns != entry['mtime_ns']:
            raise RuntimeError(f'File changed since cleanup planning: {path}')
        path.unlink()
    for relative, expected in checksums.items():
        if digest(ROOT / relative) != expected:
            raise RuntimeError(f'A protected estimation artifact changed: {relative}')
    removed_paths = {(ROOT / entry['path']).resolve() for entry in plan}
    record['reports_updated'] = []
    for name in GROUPS:
        group = ROOT / name
        if not group.is_dir():
            continue
        for path in sorted(group.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
                path.rmdir()
        if any(entry['path'].startswith(name+'/') for entry in plan):
            (group / 'CLEANUP.md').write_text(
                '# Historical output cleanup\n\n'
                'Redundant component and diagnostic plots were removed. '
                'Every stage overview, visualization metadata and key comparison image remain. '
                'All NPZ archives, final RGB, config, summary, cycles and source snapshots were preserved. '
                'Detailed image links in historical reports are marked when their figures were pruned. '
                f'The exact removal list and protected SHA-256 hashes are in `{manifest.relative_to(ROOT)}`.\n')
            readme = group / 'README.md'
            if readme.is_file():
                def update_link(match):
                    target = (group / match.group(2).split('#')[0]).resolve()
                    return (match.group(1)+'（派生图已清理）' if target in removed_paths else match.group(0))
                contents = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', update_link, readme.read_text())
                contents = '> 历史输出已精简：保留关键图和全部 NPZ，见 [清理说明](CLEANUP.md)。\n\n'+contents
                readme.write_text(contents)
                record['reports_updated'].append(str(readme.relative_to(ROOT)))
    record['completed'] = True
    manifest.write_text(json.dumps(record, indent=2))
    print('Cleanup complete; all protected estimation artifacts are byte-identical.')


if __name__ == '__main__':
    main()
