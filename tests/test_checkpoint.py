"""Checkpoint verification helpers, without Docker.

`tar_digest` is what restores are verified with (content hash at snapshot time
vs. the restored path), `fingerprint_diff` what restored and finished
environments are compared with, and `rewind.evals.corrupt` what the negative
test uses to damage a checkpoint.

    uv run python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from rewind.agent import fingerprint_diff
from rewind.env import tar_digest
from rewind.evals import corrupt


def make_tar(path: Path, files: dict[str, bytes], mtime: int = 0, mode: int = 0o644) -> Path:
    with tarfile.open(path, "w") as t:
        d = tarfile.TarInfo("root")
        d.type, d.mode = tarfile.DIRTYPE, 0o700
        t.addfile(d)
        for name, data in files.items():
            m = tarfile.TarInfo(f"root/{name}")
            m.size, m.mtime, m.mode = len(data), mtime, mode
            t.addfile(m, io.BytesIO(data))
    return path


class TarDigest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def digest(self, name: str, files: dict[str, bytes], **kw) -> str:
        return tar_digest(make_tar(self.dir / name, files, **kw))

    def test_same_content_same_digest_regardless_of_mtime_and_order(self) -> None:
        a = self.digest("a.tar", {"x": b"1", "y": b"2"}, mtime=1)
        b = self.digest("b.tar", {"y": b"2", "x": b"1"}, mtime=999)
        self.assertEqual(a, b)

    def test_content_mode_and_name_changes_are_detected(self) -> None:
        base = self.digest("a.tar", {"x": b"1"})
        self.assertNotEqual(base, self.digest("b.tar", {"x": b"2"}))
        self.assertNotEqual(base, self.digest("c.tar", {"x": b"1"}, mode=0o755))
        self.assertNotEqual(base, self.digest("d.tar", {"z": b"1"}))

    def test_corrupt_changes_the_digest_but_keeps_the_tar_readable(self) -> None:
        ckpt = self.dir / "step-003"
        ckpt.mkdir()
        tar = make_tar(ckpt / "default__root.tar", {"x": b"1", "y": b"2"})
        before = tar_digest(tar)
        corrupt(ckpt, {"step": 3})
        self.assertNotEqual(before, tar_digest(tar))
        with tarfile.open(tar) as t:
            self.assertEqual(sorted(t.getnames()), ["root", "root/x", "root/y"])


class FingerprintDiff(unittest.TestCase):
    def test_volatile_keys_are_ignored(self) -> None:
        a = {"prs": [1], "ci_runs": [{"id": 1, "state": "queued"}]}
        b = {"prs": [1], "ci_runs": [{"id": 1, "state": "success"}]}
        self.assertEqual(fingerprint_diff(a, b, ["ci_runs"]), [])
        self.assertEqual(fingerprint_diff(a, b, []), ["ci_runs"])

    def test_changed_and_missing_keys_are_reported(self) -> None:
        a = {"prs": [1], "tags": [], "clones": [{"path": "/tmp/calc"}]}
        b = {"prs": [1, 2], "tags": []}
        self.assertEqual(fingerprint_diff(a, b, []), ["clones", "prs"])

    def test_missing_fingerprint_is_unknown(self) -> None:
        self.assertIsNone(fingerprint_diff(None, {}, []))
        self.assertIsNone(fingerprint_diff(json.loads("{}"), None, []))


if __name__ == "__main__":
    unittest.main()
