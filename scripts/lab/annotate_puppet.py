"""Redraw /project1/PuppetLab's network boxes after scripts/lab/build_puppet.py: the old boxes
go, then one titled networkbox per GROUPS entry, through the td-http-api bridge. Run outside TD:

	python scripts/lab/annotate_puppet.py [port]

Prints each box's title and whether it encloses exactly its group's nodes.
"""
import ast
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PORT = sys.argv[1] if len(sys.argv) > 1 else '3031'
BASE = f'http://127.0.0.1:{PORT}'
LAB = '/project1/PuppetLab'


def call(route, method='GET', **query):
	req = urllib.request.Request(f'{BASE}{route}?{urllib.parse.urlencode(query)}', method=method,
		data=b'' if method == 'POST' else None)
	with urllib.request.urlopen(req, timeout=30) as r:
		return json.loads(r.read())


def groups():
	"""GROUPS from build_puppet.py (with HAND_MAPS, the one name it refers to)."""
	tree = ast.parse((Path(__file__).parent / 'build_puppet.py').read_text(encoding='utf-8'))
	env = {}
	for node in tree.body:
		if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) in ('HAND_MAPS', 'GROUPS'):
			exec(compile(ast.Module([node], []), 'build_puppet.py', 'exec'), env)
	return env['GROUPS']


def main():
	wanted = groups()
	old = [n['path'] for n in call('/network', path=LAB)['nodes'] if n.get('opType') == 'annotateCOMP']
	if old:
		call('/delete', 'POST', paths=','.join(old))
	for title, names in wanted.items():
		call('/annotate', 'POST', parent=LAB, paths=','.join(f'{LAB}/{n}' for n in names),
			mode='networkbox', title=title, pad=40)
	net = call('/network', path=LAB)
	for box in (n for n in net['nodes'] if n.get('opType') == 'annotateCOMP'):
		title = call('/par', path=box['path'], par='Titletext').get('value')
		enclosed = sorted(p.rsplit('/', 1)[1] for p in box.get('enclosedOPs') or [])
		print(f"{title}: {'ok' if enclosed == sorted(wanted.get(title, [])) else enclosed}")


if __name__ == '__main__':
	main()
