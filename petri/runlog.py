"""Run log for the petri experiments: notes/runs.jsonl, scratch per batch, one JSON object per line.

A batch is one tangent, idea, experiment, proof or falsification attempt. The log is not kept in git: it is
started fresh for each batch, and when the batch closes only the runs marked keep (and only their kept
parts) are saved next to the lab book, in notes/batches/, which is what gets committed.

  env/result/bin/python3 petri/runlog.py start "what this batch tries"            # fresh log; line 1 = batch header
  env/result/bin/python3 petri/runlog.py keep <id> "why it matters" [part ...]   # parts: e.g. output meter
  env/result/bin/python3 petri/runlog.py drop <id> "why it does not"
  env/result/bin/python3 petri/runlog.py tail [n]                                  # the last n lines (default 5)
  env/result/bin/python3 petri/runlog.py close [short-name]                        # save the kept parts, end the batch

Every run of mini.py, permute.py and a dish's run.py appends one line when it finishes, failed runs included,
and prints a reminder. A line's number is its id, so one run can be read on its own (ai-text-editor: read
that line). Within a batch nothing is rewritten; a mark is a new line that refers back. `start` refuses
while the previous batch is still open, so kept runs cannot be lost by starting over.

Inner runs (mini.py and permute.py running a dish's run.py themselves) set PETRI_NO_LOG=1 and are not logged
separately; the outer script's line holds their result.
"""
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, 'notes', 'runs.jsonl')
BATCHES = os.path.join(ROOT, 'notes', 'batches')
NOW = lambda: time.strftime('%Y-%m-%dT%H:%M:%S%z')


def _append(entry, log=LOG):
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, 'a') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    with open(log) as f:
        return sum(1 for _ in f)


def _lines():
    return [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []


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
    lines = _lines() if log == LOG else []
    if lines and 'batch' not in lines[0]:
        print('[run log] no batch started: begin each batch with petri/runlog.py start "what it tries"')


def write(script, record):
    """Append one run and print the reminder. Returns the run id (the line number)."""
    if os.environ.get('PETRI_NO_LOG'):
        return None
    # paths relative to the lab folder: kept parts end up in a public repository
    record = {k: (os.path.relpath(v, ROOT) if isinstance(v, str) and v.startswith(ROOT + os.sep) else v)
              for k, v in record.items()}
    entry = {'time': NOW(), 'script': script, 'argv': sys.argv[1:], 'commit': _commit(), **record}
    n = _append(entry)
    reminder(n, LOG)
    return n


def start(title, discard=False):
    lines = _lines()
    if lines and 'closed' not in lines[-1] and not discard:
        raise SystemExit('the current batch is not closed: run petri/runlog.py close [short-name] first '
                         '(or start --discard to throw it away)')
    if os.path.exists(LOG):
        os.remove(LOG)
    _append({'batch': title, 'started': NOW(), 'commit': _commit()})
    print(f'batch started: {title!r} · {os.path.relpath(LOG, os.getcwd())} line 1 is its header; runs follow')


def close(name=None):
    lines = _lines()
    if not lines:
        raise SystemExit('no run log to close')
    if 'closed' in lines[-1]:
        raise SystemExit(f'this batch is already closed (saved to {lines[-1].get("saved_to")})')
    head = lines[0] if 'batch' in lines[0] else {'batch': name or 'unnamed batch', 'started': lines[0].get('time')}
    marks = {}
    for l in lines:                                   # the latest mark for a run wins
        if 'mark' in l:
            marks[l['run']] = l
    runs = [i for i, l in enumerate(lines, 1) if 'script' in l]
    kept = []
    for rid, m in sorted(marks.items()):
        if m['mark'] != 'keep':
            continue
        run = lines[rid - 1]
        if m.get('parts'):
            run = {k: v for k, v in run.items() if k in m['parts'] or k in ('time', 'script', 'dish')}
        kept.append({'run_id': rid, 'why': m['why'], 'parts': m.get('parts', 'all'), 'run': run})
    slug = re.sub(r'[^a-z0-9]+', '-', (name or head['batch']).lower()).strip('-')[:60] or 'batch'
    os.makedirs(BATCHES, exist_ok=True)
    out = os.path.join(BATCHES, f'{time.strftime("%Y-%m-%d-%H%M")}-{slug}.jsonl')
    summary = {'batch': head['batch'], 'started': head.get('started'), 'closed': NOW(), 'runs': len(runs),
               'kept': len(kept), 'dropped': sum(1 for m in marks.values() if m['mark'] == 'drop'),
               'unmarked': len([r for r in runs if r not in marks])}
    with open(out, 'w') as f:
        for e in [summary] + kept:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    rel = os.path.relpath(out, ROOT)
    _append({'closed': NOW(), 'saved_to': rel})
    print(f'batch closed: {summary["runs"]} runs, {summary["kept"]} kept, {summary["dropped"]} dropped, '
          f'{summary["unmarked"]} unmarked -> {rel}')
    print('[run log] REMINDER: link it from the lab-book entry for this batch in notes/LABBOOK.md, then commit and push.')


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'tail'
    if cmd == 'start':
        args = [a for a in sys.argv[2:] if a != '--discard']
        if not args:
            raise SystemExit('usage: runlog.py start "what this batch tries" [--discard]')
        start(args[0], '--discard' in sys.argv)
    elif cmd == 'close':
        close(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd in ('keep', 'drop'):
        if len(sys.argv) < 4:
            raise SystemExit(f'usage: runlog.py {cmd} <id> "why" [part ...]')
        entry = {'time': NOW(), 'mark': cmd, 'run': int(sys.argv[2]), 'why': sys.argv[3]}
        if cmd == 'keep' and sys.argv[4:]:
            entry['parts'] = sys.argv[4:]
        print(f'marked run {entry["run"]} as {cmd} (log line {_append(entry)})')
    elif cmd == 'tail':
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        lines = open(LOG).read().splitlines() if os.path.exists(LOG) else []
        for i, line in enumerate(lines[-n:], start=max(1, len(lines) - n + 1)):
            print(f'{i}: {line}')
    else:
        raise SystemExit('commands: start, keep, drop, tail, close')


if __name__ == '__main__':
    main()
