"""Closed-caption TOP chain for /project1/AudioLab (POST to the bridge's /run): transcript Info
DAT -> caption_lines (Script DAT) -> caption_text (Text TOP) -> captions (Over TOP on a transparent
1920x1080 canvas) -> null_captions. Laid out on the lab's 150 x 100 grid under the transcribe band."""
import json

lab = op('/project1/AudioLab')
COL, ROW, BAND = 150, 100, -500

CALLBACKS = '''# Closed captions from the transcript Info DAT (segment, final, start_ms, end_ms, text): the
# newest sentences plus the interim guess, trimmed to the last Max Characters at a word boundary,
# cleared after Hold seconds without new text. Frame (absTime.frame) makes it cook every frame.

def onSetupParameters(scriptOp):
	page = scriptOp.appendCustomPage('Captions')
	p = page.appendInt('Maxchars', label='Max Characters')
	p[0].default = p[0].val = 90
	p[0].normMin, p[0].normMax = 20, 200
	p = page.appendFloat('Hold', label='Hold (Seconds)')
	p[0].default = p[0].val = 4.0
	p[0].normMin, p[0].normMax = 0.5, 15.0
	p = page.appendInt('Frame', label='Frame')
	p[0].expr = 'absTime.frame'
	p[0].readOnly = True

def onCook(scriptOp):
	source = scriptOp.inputs[0] if scriptOp.inputs else None
	rows = []	# (segment number, final, text)
	if source is not None and source.numRows > 1 and 'text' in [c.val for c in source.row(0)]:
		for row in range(1, source.numRows):
			text = source[row, 'text'].val.strip()
			# Skip segments with no letters or digits (Whisper emits '♪♪♪' for music and silence).
			if any(ch.isalnum() for ch in text):
				rows.append((int(source[row, 'segment'].val or 0), source[row, 'final'].val == '1', text))
	state = scriptOp.storage
	now = absTime.seconds
	signature = tuple(rows)
	newest = max((number for number, final, _ in rows if final), default=0)
	if newest < state.get('cutoff', 0):
		state['cutoff'] = 0	# transcript cleared or restarted
	if signature != state.get('signature'):
		state['signature'] = signature
		state['changed'] = now
	elif now - state.get('changed', now) > scriptOp.par.Hold.eval():
		state['cutoff'] = newest	# caption timed out: only segments after this one from now on
	cutoff = state.get('cutoff', 0)
	visible = ' '.join(text for number, final, text in rows if not final or number > cutoff)
	limit = max(1, scriptOp.par.Maxchars.eval())
	if len(visible) > limit:
		cut = visible[-limit:]
		space = cut.find(' ')
		visible = cut[space + 1:] if 0 <= space < len(cut) - 1 else cut
	scriptOp.clear()
	scriptOp.appendRow([visible])
'''

def node(op_type, name, column, row):
	n = lab.op(name) or lab.create(op_type, name)
	n.nodeX, n.nodeY = column * COL, BAND - row * ROW
	return n

lines = node('scriptDAT', 'caption_lines', 2, 0)
callbacks = lab.op('caption_lines_callbacks')
callbacks.text = CALLBACKS
lines.inputConnectors[0].connect(lab.op('transcript'))
lines.par.setuppars.pulse()
callbacks.nodeX, callbacks.nodeY = lines.nodeX, lines.nodeY - ROW
callbacks.dock = lines

text = node('textTOP', 'caption_text', 3, 0)
text.par.outputresolution = 'custom'
text.par.resolutionw, text.par.resolutionh = 1500, 190
text.par.dat = 'caption_lines'
text.par.text = ''	# else its default text is drawn as well
text.par.wordwrap = True
text.par.alignx, text.par.aligny = 'center', 'center'
text.par.fontsizex = 56
text.par.fontcolorr = text.par.fontcolorg = text.par.fontcolorb = 1.0
text.par.bgcolorr = text.par.bgcolorg = text.par.bgcolorb = 0.0
text.par.bgalpha.expr = "0.65 if op('caption_lines')[0, 0] is not None and op('caption_lines')[0, 0].val else 0"
text.par.fontalpha.expr = "1 if op('caption_lines')[0, 0] is not None and op('caption_lines')[0, 0].val else 0"

canvas = node('constantTOP', 'caption_canvas', 3, 1)
canvas.par.outputresolution = 'custom'
canvas.par.resolutionw, canvas.par.resolutionh = 1920, 1080
canvas.par.colorr = canvas.par.colorg = canvas.par.colorb = canvas.par.alpha = 0

over = node('overTOP', 'captions', 4, 0)
# Over TOP: input 1 goes over input 2, and the output takes input 2's resolution (the canvas).
over.inputConnectors[0].connect(text)
over.inputConnectors[1].connect(canvas)
over.par.prefit = 'nativeres'
over.par.justifyv = 'bottom'
over.par.justifyh = 'center'
over.par.ty = 60
over.par.tunit = 'pixels'

out = node('nullTOP', 'null_captions', 5, 0)
out.inputConnectors[0].connect(over)
out.viewer = True
print(json.dumps({'nodes': [n.name for n in (lines, text, canvas, over, out)],
	'errors': {n.name: n.errors() for n in (lines, text, canvas, over, out) if n.errors()}}))
