"""Append-only run log for the petri experiments: notes/runs.jsonl, one JSON object per line.

Every run of mini.py, permute.py and a dish's run.py appends one line at the end of the run, failed runs
included. A line's number is its id, so one run can be read on its own (ai-text-editor: read that line)
instead of scanning the whole file. Nothing is ever rewritten; a marking is a new line that refers back.

  env/result/bin/python3 petri/runlog.py keep <id> "why it matters" [part ...]   # parts: e.g. output meter
  env/result/bin/python3 petri/runlog.py drop <id> "why it does not"
  env/result/bin/python3 petri/runlog.py tail [n]                                  # the last n lines (default 5)

Inner runs (mini.py and permute.py running a dish's run.py themselves) set PETRI_NO_LOG=1 and are not logged
separately; the outer script's line holds their result.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'notes', 'runs.jsonl')


def _append(entry, log=LOG):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    with open(log) as f:
        return sum(1 for _ in f)


def _commit():
    try:
        return subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'],
                              capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def reminder(n, log):
    rel = os.path.relpath(log, os.getcwd())
    print(f'\n[run log] written: {rel} line {n} (run id {n}).')
    print('[run log] REMINDER: is this worth a lab-book entry in notes/LABBOOK.md (question, method, numbers, '
          'conclusion)? Is the log line worth keeping, whole or in part?')
    print(f'[run log]   keep: petri/runlog.py keep {n} "why" [part ...]   ·   drop: petri/runlog.py drop {n} "why"')


def write(script, record):
    """Append one run and print the reminder. Returns the run id (the line number)."""
    if os.environ.get('PETRI_NO_LOG'):
        return None
    # paths relative to the lab folder: the log is committed to a public repository
    record = {k: (os.path.relpath(v, ROOT) if isinstance(v, str) and v.startswith(ROOT + os.sep) else v)
              for k, v in record.items()}
    entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'script': script, 'argv': sys.argv[1:],
             'commit': _commit(), **record}
    n = _append(entry)
    reminder(n, LOG)
    return n


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'tail'
    if cmd in ('keep', 'drop'):
        if len(sys.argv) < 4:
            raise SystemExit(f'usage: runlog.py {cmd} <id> "why" [part ...]')
        entry = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'mark': cmd, 'run': int(sys.argv[2]), 'why': sys.argv[3]}
        if cmd == 'keep' and sys.argv[4:]:
            entry['parts'] = sys.argv[4:]
        print(f'marked run {entry["run"]} as {cmd} (log line {_append(entry)})')
    elif cmd == 'tail':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        lines = open(LOG).read().splitlines() if os.path.exists(LOG) else []
        for i, line in enumerate(lines[-n:], start=max(1, len(lines) - n + 1)):
            print(f'{i}: {line}')
    else:
        raise SystemExit('commands: keep, drop, tail')


if __name__ == '__main__':
    main()
