import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location('release', Path(__file__).resolve().parents[1] / 'scripts/release.py')
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
	def setUp(self):
		self.temp = tempfile.TemporaryDirectory()
		self.addCleanup(self.temp.cleanup)
		self.root = Path(self.temp.name)
		self.files = sorted(release.BINARY_FILES | {
			'LICENSE', 'NOTICE', 'VERSION', 'CHANGELOG.md', 'README.md', 'tox/MediaPipeHands.tox',
		})
		for name in self.files:
			path = self.root / name
			path.parent.mkdir(parents=True, exist_ok=True)
			path.write_text('fixture\n', encoding='utf-8')
		(self.root / 'VERSION').write_text('0.1.0\n', encoding='utf-8')
		(self.root / 'CHANGELOG.md').write_text('# Changes\n\n## Unreleased\n\n- First release.\n', encoding='utf-8')
		(self.root / 'release-manifest.json').write_text(json.dumps({'files': self.files}), encoding='utf-8')
		(self.root / 'scripts').mkdir()
		(self.root / 'scripts/release.py').write_bytes(Path(release.__file__).read_bytes())
		(self.root / '.gitignore').write_text('build/\nplugins/\n', encoding='utf-8')
		self.git('init', '-q')
		self.git('config', 'user.email', 'test@example.invalid')
		self.git('config', 'user.name', 'Release Test')
		self.commit()

	def git(self, *args):
		return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.STDOUT)

	def commit(self):
		self.git('add', '.')
		self.git('commit', '-qm', 'Fixture')

	def package(self):
		with contextlib.redirect_stdout(io.StringIO()):
			release.prepare(self.root, '0.1.0')
			self.commit()
			release.package(self.root)

	def test_version_bumps(self):
		self.assertEqual(release.next_version('1.2.3', 'major'), '2.0.0')
		self.assertEqual(release.next_version('1.2.3', 'minor'), '1.3.0')
		self.assertEqual(release.next_version('1.2.3', 'patch'), '1.2.4')
		for invalid in ('1.2', '../x', '01.2.3', '0.0.1'):
			with self.assertRaises(ValueError):
				release.next_version('1.2.3', invalid)

	def test_missing_license_blocks_prepare_before_writes(self):
		(self.root / 'LICENSE').unlink()
		self.commit()
		before = (self.root / 'CHANGELOG.md').read_bytes()
		with self.assertRaisesRegex(ValueError, 'LICENSE'):
			release.prepare(self.root, 'patch')
		self.assertEqual(release.version(self.root), '0.1.0')
		self.assertEqual((self.root / 'CHANGELOG.md').read_bytes(), before)

	def test_rejects_unsafe_and_unreviewed_paths(self):
		for name in ('../escape', 'C:/escape', 'models/unknown.onnx', 'media/clip.mp4',
			'cache/state', 'plugins/Other.dll', 'tox/shared/../escape.tox', 'README.md'):
			with self.subTest(name=name):
				(self.root / 'release-manifest.json').write_text(json.dumps({'files': self.files + [name]}), encoding='utf-8')
				with self.assertRaises(ValueError):
					release.manifest_files(self.root)

	def test_package_roundtrip_and_allowlist(self):
		(self.root / 'media').mkdir()
		(self.root / 'media/private.mp4').write_bytes(b'private')
		self.package()
		archive, prefix = release.artifact_paths(self.root)
		with zipfile.ZipFile(archive) as source:
			self.assertEqual(set(source.namelist()), {f'{prefix}/{name}' for name in self.files} | {f'{prefix}/release.json'})
		with contextlib.redirect_stdout(io.StringIO()):
			release.verify(self.root, tested=True)
		self.assertTrue(archive.with_suffix('.tested.json').exists())
		with self.assertRaisesRegex(ValueError, 'already exists'):
			release.package(self.root)

	def test_map_places_binaries_and_models(self):
		model = self.root / 'models/mediapipe/hand.task'
		model.parent.mkdir(parents=True)
		model.write_bytes(b'model')
		(self.root / '.gitignore').write_text('build/\nplugins/\nmodels/\n', encoding='utf-8')
		files = [name for name in self.files if name not in release.BINARY_FILES]
		mapped = {name: 'pkg/' + name.split('/')[-1] for name in release.BINARY_FILES}
		mapped['models/mediapipe/hand.task'] = 'pkg/models/mediapipe/hand.task'
		(self.root / 'release-manifest.json').write_text(json.dumps({'files': files, 'map': mapped}), encoding='utf-8')
		self.commit()
		self.package()
		archive, prefix = release.artifact_paths(self.root)
		with zipfile.ZipFile(archive) as source:
			self.assertIn(f'{prefix}/pkg/OnnxTOP.dll', source.namelist())
			self.assertEqual(source.read(f'{prefix}/pkg/models/mediapipe/hand.task'), b'model')
		with contextlib.redirect_stdout(io.StringIO()):
			release.verify(self.root)

	def test_rejects_agpl_models(self):
		model = self.root / 'models/onnx/yolo26/yolo26s.onnx'
		model.parent.mkdir(parents=True)
		model.write_bytes(b'agpl')
		manifest = {'files': self.files, 'map': {'models/onnx/yolo26/yolo26s.onnx': 'pkg/yolo26s.onnx'}}
		(self.root / 'release-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
		with self.assertRaisesRegex(ValueError, 'Excluded'):
			release.manifest_files(self.root)

	def test_archive_tampering_is_rejected(self):
		self.package()
		archive, _ = release.artifact_paths(self.root)
		with archive.open('ab') as output:
			output.write(b'tampered')
		with self.assertRaisesRegex(ValueError, 'checksum'):
			release.verify(self.root)

	def test_rebuilt_binary_invalidates_archive(self):
		self.package()
		(self.root / 'plugins/OnnxTOP.dll').write_bytes(b'rebuilt')
		self.assertEqual(self.git('status', '--porcelain', '--untracked-files=no'), b'')
		with self.assertRaisesRegex(ValueError, 'inputs changed'):
			release.verify(self.root)

	def test_new_commit_invalidates_archive(self):
		self.package()
		(self.root / 'other.txt').write_text('new commit', encoding='utf-8')
		self.commit()
		with self.assertRaisesRegex(ValueError, 'HEAD'):
			release.verify(self.root)

	def test_dirty_tree_blocks_packaging(self):
		(self.root / 'README.md').write_text('uncommitted', encoding='utf-8')
		with self.assertRaisesRegex(ValueError, 'Commit tracked changes'):
			release.package(self.root)
		self.assertFalse((self.root / 'build').exists())

	def test_empty_notes_block_prepare_before_writes(self):
		(self.root / 'CHANGELOG.md').write_text('## Unreleased\n', encoding='utf-8')
		self.commit()
		with self.assertRaisesRegex(ValueError, 'nonempty'):
			release.prepare(self.root, 'patch')
		self.assertEqual(release.version(self.root), '0.1.0')

	def test_publish_requires_clean_project_acknowledgment(self):
		self.package()
		with contextlib.redirect_stdout(io.StringIO()):
			with self.assertRaisesRegex(ValueError, 'Test the extracted ZIP'):
				release.publish(self.root, yes=True)


if __name__ == '__main__':
	unittest.main()