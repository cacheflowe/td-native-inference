"""Builds each tox's network-viewer card: docs/images/icons/<Name>.png (640 x 520, the 16:13 aspect
of a 160 x 130 COMP node), from wrappers/*.json and VERSION. Standard library + a local Chrome or
Edge for SVG -> PNG; no TouchDesigner needed. wrapper_builder.py embeds the PNG in each tox as a
locked image, so the toxes carry no renderer.

	python scripts/build_icons.py [Name ...]

The card follows td-http-api's app-icon layout (skills/td-app-icon): logo, product + version,
then rows for this tox: what it is and what it runs on. Replace docs/images/icons/logo.svg to
rebrand; colors are in COLORS.
"""

import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'images' / 'icons'
WIDTH, HEIGHT = 320, 260	# 16:13 like the node; rendered at 2x
FONT = "'Segoe UI', 'Helvetica Neue', Helvetica, Arial, sans-serif"
PRODUCT = 'td-native-inference'
COLORS = {
	'background': '#090D10',
	'surface': '#11191D',
	'surface_border': '#243239',
	'text': '#F2F5F4',
	'muted': '#91A0A5',
	'accent': '#F2A541',	# the library's accent (td-http-api uses teal)
}
ENGINES = {
	'OnnxTOP': 'ONNX \u00b7 GPU',
	'MediapipeTOP': 'MediaPipe \u00b7 CPU',
	'MediapipeaudioCHOP': 'MediaPipe audio \u00b7 CPU',
}
DEMOS = {
	'VisionDemo': ('Vision Demo', 'All video models'),
	'AudioDemo': ('Audio Demo', 'All audio models'),
}
BROWSERS = [
	Path(r'C:\Program Files\Google\Chrome\Application\chrome.exe'),
	Path(r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe'),
	Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),
	Path(r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'),
]


def fit(size, text, width):
	"""Font size for `text` to fit `width` (Segoe UI averages about 0.55 em per character)."""
	return min(size, width / max(1, len(text) * 0.55))


def card_svg(logo_svg, headline, rows):
	"""The card, laid out from the bottom up: the (label, value) rows sit a fixed padding above the
	bottom edge, then a divider and the headline (the model: what tells nodes apart at node size);
	the logo fills what's left on top."""
	s, w, c, esc = HEIGHT, WIDTH, COLORS, html.escape
	row_h, row_step, bottom_pad = s * 0.105, s * 0.125, s * 0.075
	last_row = s - bottom_pad - row_h / 2	# center of the bottom row
	first_row = last_row - row_step * (len(rows) - 1)
	divider = first_row - row_h / 2 - s * 0.045
	title_y = divider - s * 0.05	# title baseline
	logo_top, logo_bottom = s * 0.05, title_y - s * 0.105	# leftover space above the title
	m = re.search(r'<svg\b([^>]*)>(.*)</svg>', logo_svg, re.S | re.I)
	logo = ''
	if m:
		vb = re.search(r'viewBox\s*=\s*"([^"]+)"', m.group(1))
		fill = re.search(r'\bfill\s*=\s*"([^"]*)"', m.group(1), re.I)
		fill_attr = f' fill="{esc(fill.group(1), quote=True)}"' if fill else ''
		logo = (f'<svg x="{w * 0.14:.0f}" y="{logo_top:.1f}" width="{w * 0.72:.0f}" height="{logo_bottom - logo_top:.1f}" '
			f'viewBox="{vb.group(1) if vb else "0 0 100 100"}" preserveAspectRatio="xMidYMid meet"{fill_attr}>{m.group(2)}</svg>')
	body = []
	for i, (label, value, color) in enumerate(rows):
		y = first_row + i * row_step
		value_size = fit(s * 0.052, value, w * 0.60)
		body.append(
			f'<rect x="{w * 0.07:.0f}" y="{y - s * 0.0525:.1f}" width="{w * 0.86:.0f}" height="{s * 0.105:.0f}" '
			f'rx="{s * 0.025:.1f}" fill="{c["surface"]}" stroke="{c["surface_border"]}" stroke-width="{s / 256:.1f}"/>'
			f'<rect x="{w * 0.07:.0f}" y="{y - s * 0.032:.1f}" width="{w * 0.012:.0f}" height="{s * 0.064:.0f}" '
			f'rx="{s * 0.006:.1f}" fill="{c["accent"]}"/>'
			f'<text x="{w * 0.12:.0f}" y="{y + s * 0.016:.1f}" font-family="{FONT}" '
			f'font-size="{s * 0.047:.1f}" fill="{c["muted"]}">{esc(label)}</text>'
			f'<text x="{w * 0.31:.0f}" y="{y + s * 0.016:.1f}" font-family="{FONT}" '
			f'font-size="{value_size:.1f}" font-weight="600" fill="{c[color]}">{esc(value)}</text>')
	return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w * 2}" height="{s * 2}" viewBox="0 0 {w} {s}">'
		f'<rect width="{w}" height="{s}" fill="{c["background"]}"/>{logo}'
		f'<text x="{w / 2:.0f}" y="{title_y:.1f}" text-anchor="middle" font-family="{FONT}" '
		f'font-size="{fit(s * 0.085, headline, w * 0.88):.1f}" font-weight="650" fill="{c["text"]}">{esc(headline)}</text>'
		f'<line x1="{w * 0.07:.0f}" y1="{divider:.1f}" x2="{w * 0.93:.0f}" y2="{divider:.1f}" '
		f'stroke="{c["surface_border"]}" stroke-width="{s / 256:.1f}"/>{"".join(body)}</svg>')


def cards(names=None):
	"""{name: (headline, rows)} for every spec (and the demos), or just `names`. The headline is
	what tells the nodes apart at node size, so it's the model; the package is a row."""
	result = {}
	for path in sorted((ROOT / 'wrappers').glob('*.json')):
		spec = json.loads(path.read_text(encoding='utf-8'))
		engine = ENGINES.get(spec['plugin'], spec['plugin'])
		result[spec['name']] = (spec.get('label', spec['name']), [('Engine:', engine, 'accent')])
	for name, (title, what) in DEMOS.items():
		result[name] = (title, [('Models:', what, 'accent')])
	return {k: v for k, v in result.items() if not names or k in names}


def render(svg, png, browser):
	"""SVG -> PNG at the SVG's own size with a headless browser."""
	with tempfile.TemporaryDirectory() as tmp:
		page = Path(tmp) / 'card.html'
		page.write_text(f'<!doctype html><html><body style="margin:0;background:#000">{svg}</body></html>', encoding='utf-8')
		subprocess.run([str(browser), '--headless=new', '--disable-gpu', '--hide-scrollbars', '--force-device-scale-factor=1',
			f'--window-size={WIDTH * 2},{HEIGHT * 2}', f'--screenshot={png}', page.as_uri()],
			check=True, capture_output=True, timeout=60)


def main(names):
	browser = next((b for b in BROWSERS if b.exists()), None) or shutil.which('chrome') or shutil.which('msedge')
	if not browser:
		sys.exit('[build_icons] Chrome or Edge not found')
	version = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
	logo = (OUT / 'logo.svg').read_text(encoding='utf-8')
	OUT.mkdir(parents=True, exist_ok=True)
	for name, (headline, rows) in cards(names).items():
		png = OUT / f'{name}.png'
		rows = [*rows, ('Package:', f'{PRODUCT} v{version}', 'text')]
		render(card_svg(logo, headline, rows), png, browser)
		print(f'[build_icons] {png.relative_to(ROOT).as_posix()}')


if __name__ == '__main__':
	main(sys.argv[1:])
