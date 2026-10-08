"""Pin the actual pure-PyTorch upstream sources used by this adaptation."""
import hashlib

from baseline6.upstream import ROOT


def test_original_source_git_blobs():
    expected = {
        "pts/model/time_grad/epsilon_theta.py": "2ad3680c25c0f92dd236805a62d8ffdc118a1f0a",
        "pts/modules/gaussian_diffusion.py": "95ddac89c4d2a5d3aaf99934e9f932781a577e59",
    }
    for path, sha in expected.items():
        # Git checkouts may use CRLF on Windows.
        data = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert actual == sha, f"Upstream changed: review baseline adaptation for {path}"
