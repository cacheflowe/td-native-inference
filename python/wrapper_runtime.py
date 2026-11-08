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


# Sound classification wrappers. `scores` is a Lag CHOP of every class score (channel names are
# the class labels made CHOP-safe: "Child speech, kid speaking" -> Child_speech_kid_speaking).

def write_top(script_op, comp):
	"""rank, class (channel name), label, score: the Top Results highest smoothed scores."""
	scores = comp.op('scores')
	script_op.clear()
	script_op.appendRow(['rank', 'class', 'label', 'score'])
	if not scores.numChans:
		return
	values = scores.numpyArray()[:, -1]
	count = max(1, comp.par.Topcount.eval())
	for rank, index in enumerate(values.argsort()[::-1][:count], 1):
		name = scores[int(index)].name
		script_op.appendRow([rank, name, name.replace('_', ' '), f'{values[index]:.4f}'])


def write_events(script_op, comp):
	"""Per watched class: <name> (smoothed score), <name>_active (above Active Threshold) and
	<name>_onset (1 on the cook it becomes active). Unknown names are skipped."""
	scores = comp.op('scores')
	threshold = comp.par.Threshold.eval()
	previous = script_op.storage.setdefault('active', {})
	script_op.clear()
	script_op.numSamples = 1
	for name in comp.par.Watch.eval().replace(',', ' ').split():
		channel = scores[name]
		if channel is None:
			continue
		value = float(channel[channel.owner.numSamples - 1])
		active = value > threshold
		script_op.appendChan(name)[0] = value
		script_op.appendChan(name + '_active')[0] = float(active)
		script_op.appendChan(name + '_onset')[0] = float(active and not previous.get(name, False))
		previous[name] = active


def write_top_lines(script_op, comp):
	"""The debug display: one line per top result, label, bar and score."""
	top = comp.op('top')
	script_op.clear()
	lines = []
	for row in range(1, top.numRows):
		score = float(top[row, 'score'].val or 0)
		label = top[row, 'label'].val
		label = label if len(label) <= 22 else label[:21] + '…'
		# ASCII bar: block characters aren't in Consolas, and a fallback font breaks the columns.
		lines.append(f"{label:<22} {'#' * int(round(score * 10)):<10} {score:.2f}")
	script_op.appendRow(['\n'.join(lines)])


# Text to speech wrappers. `info` is the native node's Info DAT: line, state (speaking /
# queued), voice, text.

def speak_input(comp):
	"""Called every frame: speak the newest row of the wired-in DAT when it changes. The row's
	'text' column if the DAT has a header with one (e.g. MediaPipeSpeechToText's latest DAT), else
	its last non-empty cell. What's there when the wrapper starts isn't spoken (reopening a
	project doesn't replay the last line); each new text is spoken once."""
	if not comp.par.Speakinput.eval():
		return
	source = comp.op('input')
	text = ''
	if source.numRows:
		header = [cell.val for cell in source.row(0)]
		if 'text' in header:
			text = source[source.numRows - 1, 'text'].val if source.numRows > 1 else ''
		else:
			cells = [cell.val for cell in source.row(source.numRows - 1) if cell.val.strip()]
			text = cells[-1] if cells else ''
	text = text.strip()
	if 'spoken_input' not in comp.storage:
		comp.storage['spoken_input'] = text	# already there at startup: not new
		return
	if not text or text == comp.storage['spoken_input']:
		return
	comp.storage['spoken_input'] = text
	comp.par.Text = text
	if not comp.par.Speakonchange.eval():	# else the new Text is spoken already
		comp.op('inference').par.Speak.pulse()


def write_speaking(script_op, comp):
	"""The line being spoken, for the caption."""
	info = comp.op('info')
	text = ''
	if info.numRows > 1 and [cell.val for cell in info.row(0)][:2] == ['line', 'state']:
		text = next((info[row, 'text'].val for row in range(1, info.numRows) if info[row, 'state'].val == 'speaking'), '')
	script_op.clear()
	script_op.appendRow([text])


# Speech to text wrappers. The transcript Info DAT has columns segment (running number), final (1/0),
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
	hold = comp.par.Hold.eval()
	if now - state.get('cooked', -1e9) > hold:
		# Not shown for a while (or never): what arrived meanwhile is old news, not a caption.
		state.update(cutoff=newest, signature=signature, changed=now)
	state['cooked'] = now
	if newest < state.get('cutoff', 0):
		state['cutoff'] = 0	# transcript cleared or restarted
	if signature != state.get('signature'):
		gap = comp.par.Newcaption.eval()
		if gap > 0 and now - state.get('changed', now) > gap:
			state['cutoff'] = state.get('newest', 0)	# new speech after a quiet gap: fresh caption
		state['signature'] = signature
		state['changed'] = now
	elif now - state.get('changed', now) > hold:
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
