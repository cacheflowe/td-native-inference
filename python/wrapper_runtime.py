"""Runtime support embedded in every generated model wrapper (the wrapper's `component` DAT).

Only what a wrapper needs while it runs. Building and exporting live in wrapper_builder.py.
"""

import re


def write_channels(script_op, comp):
	"""Per-track numeric columns (t{id}_{column}) plus the native node's Info CHOP stats."""
	script_op.clear()
	script_op.numSamples = 1
	tracks = comp.op('tracks')
	tracks.cook()
	if tracks.numRows:
		header = [cell.val for cell in tracks.row(0)]
		if 'track_id' in header:
			id_index = header.index('track_id')
			for row in tracks.rows()[1:]:
				track_id = row[id_index].val
				for name, cell in zip(header, row):
					try:
						value = float(cell.val)
					except ValueError:
						continue
					channel = re.sub(r'[^a-zA-Z0-9_]', '_', f't{track_id}_{name}')
					script_op.appendChan(channel)[0] = value
	stats = comp.op('inference_stats')
	stats.cook(force=True)
	for channel in stats.chans():
		script_op.appendChan(channel.name)[0] = channel[0]
