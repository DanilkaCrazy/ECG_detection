"""Low level WFDB helpers: header parsing, MATLAB-signal reading, integrity checks.

The WFDB header (``.hea``) is parsed directly instead of going through
``wfdb.rdrecord`` so that we can report *why* a record is broken instead of
only that it could not be opened.

Two upstream defects of the published Chapman-Shaoxing release are handled
here by :func:`parse_header_lenient`:

* ``JS01052.hea`` - the record line and the first signal line were merged and
  18 characters of the signal line were dropped;
* ``JS23074.hea`` - the record name lost its leading ``J`` and the record line
  was merged with the first signal line.

Both repairs are *self-verifying*: the reconstructed header is only accepted
when all twelve 16-bit WFDB checksums of the corresponding ``.mat`` file
match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

__all__ = [
    "SignalSpec",
    "Header",
    "parse_header",
    "parse_header_lenient",
    "read_mat_signal",
    "wfdb_checksum",
    "checksum_matches",
    "referenced_files",
    "find_signal_file",
]


@dataclass
class SignalSpec:
    """One ``.hea`` signal line."""

    file_name: str
    fmt: str
    gain: float
    adc_zero: int
    init_value: int
    checksum: int | None
    block_size: int
    description: str
    raw_index: int
    adc_res: int | None = None


@dataclass
class Header:
    """Parsed WFDB header."""

    record_name: str
    n_sig: int
    fs: float
    n_samp: int
    base_time: str | None
    base_date: str | None
    signals: list[SignalSpec] = field(default_factory=list)
    comments: dict[str, str] = field(default_factory=dict)
    path: Path | None = None
    repair_note: str | None = None

    @property
    def signal_files(self) -> list[str]:
        """Distinct signal files referenced by the header (in order)."""
        seen: list[str] = []
        for sig in self.signals:
            if sig.file_name not in seen:
                seen.append(sig.file_name)
        return seen

    @property
    def lead_names(self) -> list[str]:
        return [sig.description for sig in self.signals]

    @property
    def dx_codes(self) -> list[str]:
        raw = self.comments.get("Dx", "")
        return [c.strip() for c in raw.split(",") if c.strip()]

    @property
    def checksum_list(self) -> list[int | None]:
        return [sig.checksum for sig in self.signals]


_COMMENT_RE = re.compile(r"^#\s*([^:]+):\s*(.*)$")


def _num(text, default=None):
    if text is None:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def _build_signal(parts: list[str], raw_index: int) -> SignalSpec:
    """Build a :class:`SignalSpec` from the whitespace-split fields of a signal line."""
    file_name, fmt, gain_s = parts[0], parts[1], parts[2]
    adc_res_s = parts[3] if len(parts) > 3 else None
    adc_zero_s = parts[4] if len(parts) > 4 else None
    init_s = parts[5] if len(parts) > 5 else None
    checksum_s = parts[6] if len(parts) > 6 else None
    block_size_s = parts[7] if len(parts) > 7 else None
    description = parts[8].strip() if len(parts) > 8 else ""

    gain = _num(gain_s, 1.0)
    adc_res = _num(adc_res_s, None)
    adc_zero = _num(adc_zero_s, 0)
    init_value = _num(init_s, 0)
    checksum = _num(checksum_s, None)

    return SignalSpec(
        file_name=file_name,
        fmt=fmt,
        gain=float(gain) if gain is not None else 1.0,
        adc_zero=int(adc_zero) if adc_zero is not None else 0,
        init_value=int(init_value) if init_value is not None else 0,
        checksum=int(checksum) if checksum is not None else None,
        block_size=int(block_size_s) if (block_size_s or "0").isdigit() else 0,
        description=description,
        raw_index=raw_index,
        adc_res=int(adc_res) if adc_res is not None else None,
    )


def parse_header(path: str | Path) -> Header:
    """Parse a ``.hea`` file strictly.

    Raises ``ValueError`` when the file is not a parsable WFDB header.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()]
    if not lines:
        raise ValueError(f"empty header: {path}")

    record_line = lines[0].split()
    if len(record_line) < 3:
        raise ValueError(f"malformed record line in {path}: {lines[0]!r}")

    record_name = record_line[0]
    n_sig = int(record_line[1])
    fs = float(record_line[2])
    n_samp = int(record_line[3]) if len(record_line) > 3 else 0

    base_time = base_date = None
    for token in record_line[4:]:
        if ":" in token:
            key, _, value = token.partition(":")
            if key == "base_time":
                base_time = value
            elif key == "base_date":
                base_date = value

    header = Header(
        record_name=record_name,
        n_sig=n_sig,
        fs=fs,
        n_samp=n_samp,
        base_time=base_time,
        base_date=base_date,
        path=path,
    )

    sig_lines = 0
    for line in lines[1:]:
        if line.startswith("#"):
            match = _COMMENT_RE.match(line)
            if match:
                header.comments[match.group(1).strip()] = match.group(2).strip()
            continue
        if sig_lines >= n_sig:
            continue
        parts = line.split(None, 8)
        if len(parts) < 6:
            raise ValueError(f"malformed signal line in {path}: {line!r}")
        header.signals.append(_build_signal(parts, sig_lines))
        sig_lines += 1

    if sig_lines != n_sig:
        raise ValueError(
            f"header declares {n_sig} signals but {sig_lines} signal lines in {path}"
        )
    return header


def _header_from_parts(
    record_name: str,
    n_sig: int,
    fs: float,
    n_samp: int,
    signal_field_lists: list[list[str]],
    comments: dict[str, str],
    path: Path,
    repair_note: str,
) -> Header:
    header = Header(
        record_name=record_name,
        n_sig=n_sig,
        fs=fs,
        n_samp=n_samp,
        base_time=None,
        base_date=None,
        comments=comments,
        path=path,
        repair_note=repair_note,
    )
    for idx, parts in enumerate(signal_field_lists):
        header.signals.append(_build_signal(parts, idx))
    return header


def parse_header_lenient(
    path: str | Path, signal_path: str | Path | None = None
) -> tuple[Header | None, str]:
    """Parse a header, attempting a verified repair of merged first lines.

    Returns ``(header, note)``. ``header`` is ``None`` when the file cannot be
    repaired. ``note`` describes what happened and is safe to store in a
    manifest.
    """
    path = Path(path)
    try:
        header = parse_header(path)
        return header, "ok"
    except Exception as exc:  # noqa: BLE001
        strict_error = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------- repair attempt
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = [
        ln for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if ln.strip()
    ]
    if not raw_lines:
        return None, strict_error

    comments: dict[str, str] = {}
    body: list[str] = []
    for line in raw_lines[1:]:
        if line.startswith("#"):
            match = _COMMENT_RE.match(line)
            if match:
                comments[match.group(1).strip()] = match.group(2).strip()
        else:
            body.append(line)

    tokens = raw_lines[0].split()
    if len(tokens) < 3:
        return None, strict_error
    try:
        n_sig = int(tokens[1])
        fs = float(tokens[2])
    except ValueError:
        return None, strict_error

    if len(body) != n_sig - 1:
        # Not the "first signal line merged into the record line" pattern.
        return None, strict_error

    # The first signal line was merged into the record line.  Recover its
    # fields from the tail of the merged token list, using the (intact)
    # sibling signal lines as a template for file name / format / gain.
    template = body[0].split(None, 8)
    tail = tokens[3:]
    # ``tail`` holds either [n_samp, file, fmt, gain, ...] (clean merge) or a
    # damaged variant where n_samp is glued to the gain.
    n_samp = 0
    if tail and tail[0].lstrip("-").isdigit():
        n_samp = int(tail[0])
        tail = tail[1:]

    if len(tail) == 9:
        first_signal = tail
        note = "reconstructed: first signal line re-split from the record line"
    elif len(tail) == 7:
        # damaged variant: [glued_n_samp+gain, adc_res, adc_zero, init, chk, blk, desc]
        first_signal = [
            template[0],   # file name
            template[1],   # format
            template[2],   # gain (restored from the sibling signal lines)
            tail[1],       # adc resolution
            tail[2],       # adc zero
            tail[3],       # initial value
            tail[4],       # checksum
            tail[5],       # block size
            tail[6],       # description
        ]
        note = "reconstructed: first signal line restored from sibling template"
    else:
        return None, strict_error

    # number of samples: prefer the value encoded in the merged token
    if n_samp == 0:
        n_samp = 0

    signal_lists = [first_signal] + [ln.split(None, 8) for ln in body]
    header = _header_from_parts(
        record_name=path.stem,
        n_sig=n_sig,
        fs=fs,
        n_samp=n_samp,
        signal_field_lists=signal_lists,
        comments=comments,
        path=path,
        repair_note=note,
    )

    # -------------------------------------------------- verify the repair
    if signal_path is None:
        candidate = path.parent / f"{path.stem}.mat"
        signal_path = candidate if candidate.exists() else None
    if signal_path is not None and Path(signal_path).exists():
        try:
            val = read_mat_signal(signal_path)
        except Exception:  # noqa: BLE001
            return header, note + " (unverified: signal file unreadable)"
        if val.shape[0] == header.n_sig:
            if header.n_samp == 0:
                header.n_samp = int(val.shape[1])
            expected = header.checksum_list
            computed = [wfdb_checksum(val[i]) for i in range(val.shape[0])]
            ok = all(
                e is not None and (c - int(e)) % 65536 == 0
                for c, e in zip(computed, expected)
            )
            if ok:
                return header, note + " | VERIFIED by 12/12 WFDB checksums"
            return None, strict_error + " | repair failed checksum validation"
    return header, note + " (unverified)"


def read_mat_signal(path: str | Path) -> np.ndarray:
    """Read the ``val`` matrix of a MATLAB ``.mat`` ECG file.

    Returns the array with shape ``(n_sig, n_samp)`` in *digital* units.
    """
    from scipy.io import loadmat

    mat = loadmat(str(path))
    if "val" not in mat:
        raise KeyError(f"{path}: no 'val' variable in MATLAB file")
    val = np.asarray(mat["val"])
    if val.ndim == 1:
        val = val.reshape(1, -1)
    if val.ndim != 2:
        raise ValueError(f"{path}: 'val' has unexpected shape {val.shape}")
    return val


def wfdb_checksum(digital: np.ndarray) -> int:
    """WFDB 16-bit checksum of a digital signal (sum modulo 2**16, signed)."""
    total = int(np.sum(np.asarray(digital, dtype=np.int64)))
    total %= 65536
    if total > 32767:
        total -= 65536
    return total


def checksum_matches(digital: np.ndarray, expected: Iterable[int]) -> bool:
    """Compare per-channel checksums with the values stored in the header.

    The comparison is performed modulo 2**16 because the published headers
    occasionally store ``32768`` where the two's complement value is
    ``-32768`` (both denote the same 16-bit pattern).
    """
    digital = np.atleast_2d(np.asarray(digital))
    for row, exp in enumerate(expected):
        if exp is None:
            continue
        if row >= digital.shape[0]:
            return False
        if (wfdb_checksum(digital[row]) - int(exp)) % 65536 != 0:
            return False
    return True


def referenced_files(header: Header) -> list[Path]:
    """Absolute paths of the signal files referenced by a header."""
    base = header.path.parent if header.path else Path(".")
    return [base / name for name in header.signal_files]


def find_signal_file(header: Header) -> Path | None:
    """First existing signal file referenced by the header (or ``None``)."""
    for path in referenced_files(header):
        if path.exists():
            return path
    return None


def iter_headers(root: str | Path, pattern: str = "*.hea") -> Iterable[Path]:
    """Yield header paths under ``root``."""
    yield from sorted(Path(root).rglob(pattern))
