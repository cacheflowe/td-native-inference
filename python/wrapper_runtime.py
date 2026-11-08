"""Runtime support embedded in every generated model wrapper (the wrapper's `component` DAT).

Only what a wrapper needs while it runs. Building and exporting live in wrapper_builder.py.
"""

import re

import td	# imported modules don't get TD's globals (absTime)


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


# Speech wrappers. The transcript Info DAT has columns segment (running number), final (1/0),
# start_ms, end_ms, text: final sentences oldest first, then the interim guess if there is one.

def _transcript_rows(transcript):
	"""(segment, final, start_ms, end_ms, text) for rows with any letter or digit (Whisper writes
	'♪♪' for music and silence)."""
	rows = []
	if transcript.numRows > 1 and [cell.val for cell in transcript.row(0)][:1] == ['segment']:
		for row in range(1, transcript.numRows):
			text = transcript[row, 'text'].val.strip()
			if any(ch.isalnum() for ch in text):
				rows.append((int(transcript[row, 'segment'].val or 0), transcript[row, 'final'].val == '1',
					transcript[row, 'start_ms'].val, transcript[row, 'end_ms'].val, text))
	return rows


def write_latest(script_op, comp):
	"""One row: the newest final sentence and the current interim guess."""
	rows = _transcript_rows(comp.op('transcript'))
	final = next((row for row in reversed(rows) if row[1]), None)
	interim = next((row[4] for row in reversed(rows) if not row[1]), '')
	script_op.clear()
	script_op.appendRow(['segment', 'start_ms', 'end_ms', 'text', 'interim'])
	script_op.appendRow([final[0], final[2], final[3], final[4], interim] if final else ['', '', '', '', interim])


def write_captions(script_op, comp):
	"""Closed captions: the newest sentences plus the interim guess, trimmed to the last Max
	Characters at a word boundary. Text arriving after New Caption After seconds of quiet starts a
	fresh caption; the caption clears after Hold seconds without new text. Must cook every frame."""
	rows = [(number, final, text) for number, final, _, _, text in _transcript_rows(comp.op('transcript'))]
	state = script_op.storage
	now = td.absTime.seconds
	signature = tuple(rows)
	newest = max((number for number, final, _ in rows if final), default=0)
	if newest < state.get('cutoff', 0):
		state['cutoff'] = 0	# transcript cleared or restarted
	if signature != state.get('signature'):
		gap = comp.par.Newcaption.eval()
		if gap > 0 and now - state.get('changed', now) > gap:
			state['cutoff'] = state.get('newest', 0)	# new speech after a quiet gap: fresh caption
		state['signature'] = signature
		state['changed'] = now
	elif now - state.get('changed', now) > comp.par.Hold.eval():
		state['cutoff'] = newest	# caption timed out: only segments after this one from now on
	state['newest'] = newest
	cutoff = state.get('cutoff', 0)
	visible = ' '.join(text for number, final, text in rows if not final or number > cutoff)
	limit = max(1, comp.par.Maxchars.eval())
	if len(visible) > limit:
		cut = visible[-limit:]
		space = cut.find(' ')
		visible = cut[space + 1:] if 0 <= space < len(cut) - 1 else cut
	script_op.clear()
	script_op.appendRow([visible])
