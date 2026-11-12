"""Static build of the petri viewer: petri/static/index.html plus one trace file per fixed sample.

  env/result/bin/python3 petri/build_static.py

The page is viewer.html with the sample list injected; it fetches traces/<name>.b64.txt relative to itself,
so a phone only downloads the sample it opens. Publish index.html with traces/*.b64.txt as supporting files.
"""
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace import HERE, Tracer  # noqa: E402  (petri/trace.py)

OUT = os.path.join(HERE, 'static')


def main():
    prompts = json.load(open(os.path.join(HERE, 'prompts.json')))['prompts']
    os.makedirs(os.path.join(OUT, 'traces'), exist_ok=True)
    t = Tracer()
    lst = []
    for p in prompts:
        data = t.run(p['text'], p['name'], p['domain'], 'petri/prompts.json (fixed sample)')
        # The artifact host serves no binary types, so traces ship as base64 text (.b64.txt, text/plain).
        rel = f"traces/{p['name']}.b64.txt"
        open(os.path.join(OUT, rel), 'w').write(base64.b64encode(data).decode())
        lst.append({'name': p['name'], 'domain': p['domain'], 'text': p['text'], 'url': rel})
        print(f"{rel}: {len(data) / 1e6:.2f} MB", flush=True)
    w = t.weights()
    open(os.path.join(OUT, 'traces/weights.b64.txt'), 'w').write(base64.b64encode(w).decode())
    print(f"traces/weights.b64.txt: {len(w) / 1e6:.2f} MB", flush=True)
    cfg = {'mode': 'static', 'prompts': lst, 'weights': 'traces/weights.b64.txt'}
    inject = '<script>window.PETRI_STATIC = ' + json.dumps(cfg) + ';</script>\n'
    page = open(os.path.join(HERE, 'viewer.html')).read()
    marker = '<!--PETRI_STATIC-->'
    if marker not in page:
        raise SystemExit(f'viewer.html has no {marker} marker')
    open(os.path.join(OUT, 'index.html'), 'w').write(page.replace(marker, inject, 1))
    print('wrote', os.path.join(OUT, 'index.html'), f'(blocks: {t.blocks_meta["kind"]})')


if __name__ == '__main__':
    main()
