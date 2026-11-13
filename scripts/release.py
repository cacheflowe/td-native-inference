"""Prepare versions, package a committed tree, verify, and publish releases.

release-manifest.json lists the archive's contents: "files" are copied to the same path, "map"
copies a source path to a different archive path (the plugin folder users copy into
Documents/Derivative/Plugins). Built binaries and redistributable model files are untracked; their
hashes are recorded. Models must be listed one by one from models/; AGPL models never ship.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
VERSION_PATTERN = r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)'
BINARY_FILES = {
	'plugins/OnnxTOP.dll', 'plugins/onnxtop_tool.exe',
	'plugins/MediaPipeTOP.dll', 'plugins/libmediapipe.dll', 'plugins/MediaPipeCHOP.dll', 'plugins/TrackerDAT.dll', 'plugins/PuppetCHOP.dll',
}
# Our plugins compile in "td-native-inference <VERSION>" (CMake reads VERSION); packaging refuses
# a plugin built for another version.
VERSIONED_PLUGINS = {'plugins/OnnxTOP.dll', 'plugins/MediaPipeTOP.dll', 'plugins/MediaPipeCHOP.dll', 'plugins/TrackerDAT.dll',
	'plugins/PuppetCHOP.dll'}
MODEL_SUFFIXES = ('.onnx', '.task', '.tflite', '.litertlm')
# Licenses that don't allow bundling in this Apache-2.0 package (fetch_models.ps1 gets them).
BLOCKED_MODELS = ('yolo', 'rvm')


def fail(message):
	raise ValueError(message)


def git(root, *arguments):
	return subprocess.check_output(['git', *arguments], cwd=root).decode('utf-8').strip()


def clean_commit(root):
	commit = git(root, 'rev-parse', 'HEAD')
	if git(root, 'status', '--porcelain', '--untracked-files=no'):
		fail('Commit tracked changes before continuing.')
	return commit


def version(root):
	value = (root / 'VERSION').read_text(encoding='utf-8').strip()
	if not re.fullmatch(VERSION_PATTERN, value):
		fail('VERSION must contain X.Y.Z with no leading zeroes.')
	return value


def next_version(current, bump):
	parts = [int(part) for part in current.split('.')]
	if bump in ('major', 'minor', 'patch'):
		index = ('major', 'minor', 'patch').index(bump)
		parts[index] += 1
		parts[index + 1:] = [0] * (2 - index)
		return '.'.join(map(str, parts))
	if not re.fullmatch(VERSION_PATTERN, bump):
		fail('Use major, minor, patch, or X.Y.Z.')
	if tuple(map(int, bump.split('.'))) < tuple(parts):
		fail('Cannot release an older version.')
	return bump


def _safe_path(name):
	if not isinstance(name, str):
		fail('Manifest paths must be strings.')
	path = PurePosixPath(name)
	if (path.is_absolute() or path.as_posix() != name or '..' in path.parts
		or '\\' in name or ':' in name or not path.parts):
		fail(f'Unsafe manifest path: {name}')
	return path


def is_untracked_input(name):
	"""Built binaries and model files: gitignored, so hashed instead of checked against HEAD."""
	return name in BINARY_FILES or (name.startswith('models/') and name.lower().endswith(MODEL_SUFFIXES))


def manifest_files(root):
	"""[(source, archive path)], sorted by archive path."""
	manifest = json.loads((root / 'release-manifest.json').read_text(encoding='utf-8'))
	files, mapped = manifest.get('files'), manifest.get('map', {})
	if not isinstance(files, list) or not files or not isinstance(mapped, dict):
		fail('release-manifest.json needs a nonempty files list (and map, if any, an object).')
	entries = [(name, name) for name in files] + list(mapped.items())
	seen = set()
	for source, target in entries:
		path = _safe_path(source)
		_safe_path(target)
		if target.casefold() in seen:
			fail(f'Duplicate manifest path: {target}')
		seen.add(target.casefold())
		suffix = path.suffix.lower()
		model = source.startswith('models/') and suffix in MODEL_SUFFIXES
		if (source.startswith(('media/', 'build/', 'cache/'))
			or any(part in ('trt_cache', '__pycache__', '.git') for part in path.parts)
			or (source.startswith('models/') and source != 'models/README.md' and not model)
			or (model and any(blocked in source.lower() for blocked in BLOCKED_MODELS))
			or (suffix in MODEL_SUFFIXES and not model)
			or (suffix in ('.dll', '.exe') and source not in BINARY_FILES)
			or suffix in ('.engine', '.toe')):
			fail(f'Excluded release content: {source}')
		if root.resolve() not in (root / source).resolve().parents:
			fail(f'Manifest path escapes the repository: {source}')
	sources = {source for source, _ in entries}
	targets = {target for _, target in entries}
	required = {'LICENSE', 'NOTICE', 'VERSION', 'CHANGELOG.md', 'README.md'}
	if not BINARY_FILES.issubset(sources) or not required.issubset(targets) or not any(
		target.startswith('tox/') and not target.startswith('tox/shared/') and target.endswith('.tox')
		for target in targets
	):
		fail('Manifest must include licenses, version/docs, all runtime binaries, and a model tox.')
	missing = [source for source, _ in entries if not (root / source).is_file()]
	if missing:
		fail('Release prerequisites missing:\n  ' + '\n  '.join(missing))
	for name in ('LICENSE', 'NOTICE'):
		if not (root / name).read_text(encoding='utf-8').strip():
			fail(f'{name} must not be empty.')
	return sorted(entries, key=lambda entry: entry[1])


def notes_section(text, heading):
	match = re.search(r'^## ' + re.escape(heading) + r'[ \t]*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
	if not match or not match.group(1).strip():
		fail(f'CHANGELOG.md needs a nonempty "## {heading}" section.')
	return match


def prepare(root, bump):
	clean_commit(root)
	manifest_files(root)
	new = next_version(version(root), bump)
	changelog = root / 'CHANGELOG.md'
	text = changelog.read_text(encoding='utf-8')
	if re.search(r'^## v' + re.escape(new) + r'(?: |$)', text, re.M):
		fail(f'Changelog already contains v{new}.')
	match = notes_section(text, 'Unreleased')
	heading = f'v{new} ({datetime.date.today().isoformat()})'
	replacement = f'## Unreleased\n\n## {heading}\n\n{match.group(1).strip()}\n\n'
	(root / 'VERSION').write_text(new + '\n', encoding='utf-8')
	changelog.write_text((text[:match.start()] + replacement + text[match.end():]).rstrip() + '\n', encoding='utf-8')
	print(f'[release] Prepared v{new}. Review and commit VERSION + CHANGELOG.md, then run package.')


def sha256(data):
	return hashlib.sha256(data).hexdigest()


def artifact_paths(root):
	name = f'td-native-inference-v{version(root)}'
	return root / 'build' / f'{name}.zip', name


def tracked_inputs(root, files):
	for name in ['release-manifest.json', 'scripts/release.py', *(source for source, _ in files)]:
		if is_untracked_input(name):
			continue
		git(root, 'ls-files', '--error-unmatch', '--', name)
		committed = subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=root)
		working = (root / name).read_bytes()
		if committed.replace(b'\r\n', b'\n') != working.replace(b'\r\n', b'\n'):
			fail(f'File does not match HEAD: {name}')


def release_notes(root):
	text = (root / 'CHANGELOG.md').read_text(encoding='utf-8')
	match = re.search(r'^## (v' + re.escape(version(root)) + r' \(\d{4}-\d{2}-\d{2}\))$', text, re.M)
	if not match:
		fail('Run prepare and commit the release changelog first.')
	return notes_section(text, match.group(1)).group(1).strip() + '\n'


def package(root):
	commit = clean_commit(root)
	files = manifest_files(root)
	tracked_inputs(root, files)
	notes = release_notes(root)
	archive, prefix = artifact_paths(root)
	if archive.exists():
		fail(f'{archive.name} already exists; use a new version or remove the local artifact yourself.')
	payload = {target: (root / source).read_bytes() for source, target in files}
	marker = f'td-native-inference {version(root)}'.encode()
	for source, target in files:
		if source in VERSIONED_PLUGINS and marker not in payload[target]:
			fail(f'{source} was not built for v{version(root)}: rebuild it from this commit.')
	metadata = {
		'version': version(root), 'source_commit': commit,
		'files': {name: sha256(data) for name, data in payload.items()},
		'binary_provenance': 'Local build; hashes recorded, not proof of a reproducible build.',
	}
	archive.parent.mkdir(exist_ok=True)
	partial = archive.with_suffix('.zip.tmp')
	try:
		with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_DEFLATED) as output:
			for name, data in payload.items():
				output.writestr(f'{prefix}/{name}', data)
			output.writestr(f'{prefix}/release.json', json.dumps(metadata, indent=2) + '\n')
		if clean_commit(root) != commit:
			fail('HEAD changed during packaging.')
		partial.rename(archive)
	finally:
		partial.unlink(missing_ok=True)
	archive.with_suffix('.sha256').write_text(f'{sha256(archive.read_bytes())}  {archive.name}\n', encoding='utf-8')
	archive.with_suffix('.notes.md').write_text(notes, encoding='utf-8')
	print(f'[release] Packaged {archive} from {commit}. Extract and test in a fresh TD project.')


def verify(root, tested=False):
	commit = clean_commit(root)
	files = manifest_files(root)
	tracked_inputs(root, files)
	archive, prefix = artifact_paths(root)
	digest = sha256(archive.read_bytes())
	expected = f'{digest}  {archive.name}\n'
	if archive.with_suffix('.sha256').read_text(encoding='utf-8') != expected:
		fail('Archive checksum mismatch.')
	with zipfile.ZipFile(archive) as source:
		names = source.namelist()
		wanted = {f'{prefix}/{target}' for _, target in files} | {f'{prefix}/release.json'}
		if len(names) != len(wanted) or set(names) != wanted:
			fail('Archive contents do not match the manifest.')
		metadata = json.loads(source.read(f'{prefix}/release.json'))
		if metadata.get('version') != version(root) or metadata.get('source_commit') != commit:
			fail('Archive does not belong to this version and HEAD. Package again from the release commit.')
		actual = {target: sha256(source.read(f'{prefix}/{target}')) for _, target in files}
		if metadata.get('files') != actual:
			fail('Archive file hashes do not match release.json.')
		if actual != {target: sha256((root / name).read_bytes()) for name, target in files}:
			fail('Release inputs changed after packaging.')
	if archive.with_suffix('.notes.md').read_text(encoding='utf-8') != release_notes(root):
		fail('Release notes changed after packaging.')
	receipt = archive.with_suffix('.tested.json')
	if tested:
		receipt.write_text(json.dumps({'sha256': digest, 'source_commit': commit}, indent=2) + '\n', encoding='utf-8')
	print(f'[release] Verified {archive.name}. Clean-project test acknowledgment: {receipt.exists()}')
	return archive, digest, commit


def publish(root, yes=False):
	archive, digest, commit = verify(root)
	receipt = archive.with_suffix('.tested.json')
	if not receipt.is_file() or json.loads(receipt.read_text(encoding='utf-8')) != {
		'sha256': digest, 'source_commit': commit,
	}:
		fail('Test the extracted ZIP, then run verify --tested-in-clean-project.')
	tag = f'v{version(root)}'
	if git(root, 'tag', '--list', tag) or git(root, 'ls-remote', '--tags', 'origin', f'refs/tags/{tag}'):
		fail(f'Tag {tag} already exists; do not reuse release versions.')
	subprocess.run(['gh', 'auth', 'status'], cwd=root, check=True)
	origin = git(root, 'remote', 'get-url', 'origin')
	commands = [
		['git', 'tag', '-a', tag, commit, '-m', tag],
		['git', 'push', 'origin', f'refs/tags/{tag}'],
		['gh', 'release', 'create', tag, str(archive), str(archive.with_suffix('.sha256')),
			'--repo', origin, '--verify-tag', '--title', tag,
			'--notes-file', str(archive.with_suffix('.notes.md'))],
	]
	for command in commands:
		print('  ' + subprocess.list2cmdline(command))
	if not yes and input('Publish this release? [y/N] ').strip().lower() != 'y':
		fail('Cancelled.')
	for command in commands:
		subprocess.run(command, cwd=root, check=True)


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	commands = parser.add_subparsers(dest='command', required=True)
	commands.add_parser('check', help='read-only packaging prerequisite check')
	prepare_parser = commands.add_parser('prepare', help='bump VERSION and roll the changelog')
	prepare_parser.add_argument('bump')
	commands.add_parser('package', help='package the committed release tree')
	verify_parser = commands.add_parser('verify', help='verify archive integrity and source identity')
	verify_parser.add_argument('--tested-in-clean-project', action='store_true')
	publish_parser = commands.add_parser('publish', help='tag, push, and create a GitHub release')
	publish_parser.add_argument('--yes', action='store_true')
	args = parser.parse_args()
	try:
		if args.command == 'check':
			files = manifest_files(ROOT)
			print(f'[release] v{version(ROOT)}: {len(files)} release inputs present. This does not certify portability or licenses.')
		elif args.command == 'prepare':
			prepare(ROOT, args.bump)
		elif args.command == 'package':
			package(ROOT)
		elif args.command == 'verify':
			verify(ROOT, args.tested_in_clean_project)
		elif args.command == 'publish':
			publish(ROOT, args.yes)
	except (ValueError, OSError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
		print(f'[release] {error}', file=sys.stderr)
		return 1
	return 0


if __name__ == '__main__':
	sys.exit(main())