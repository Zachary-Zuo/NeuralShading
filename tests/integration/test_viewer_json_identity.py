import json
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys

import pytest

from ncls.core.identity import canonical_json

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("cmake") is None, reason="Windows C++ toolchain required")
def test_viewer_json_identity_matches_python_double_and_unicode(tmp_path: Path):
    build = ROOT / "artifacts/tests/viewer-identity"
    subprocess.run(["cmake", "-S", str(ROOT / "tests/cpp/viewer_identity"), "-B", str(build), "-G", "Visual Studio 17 2022", "-A", "x64"], check=True, capture_output=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release"], check=True, capture_output=True)
    rng = random.Random(953)
    values = [0.0, -0.0, 1e-5, 1e-4, 1e15, 1e16, 0.10000000149011612, 2e-5]
    for _ in range(4096):
        bits = rng.getrandbits(64)
        if (bits >> 52) & 2047 != 2047:
            values.append(struct.unpack("<d", struct.pack("<Q", bits))[0])
    document = {"中文": "法线/阴影\n", "numbers": values, "integer": [2**64-1, -(2**63)], "array": [True, False, None]}
    path = tmp_path / "identity.json"
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    result = subprocess.run([str(build / "Release/viewer_identity.exe"), str(path)], check=True, capture_output=True)
    assert result.stdout.decode("utf-8") == canonical_json(document)
