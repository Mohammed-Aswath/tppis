"""Fetch scripts for the two real datasets in Section 5 of the paper.

Neither dataset is redistributed. These helpers download from the public
sources the paper cites, cache locally, and raise a clear error if a source
is unavailable.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("TPPIS_DATA", Path.home() / ".tppis_data"))

# Sources as cited in Tanaka and Matsui (2023), Section 5.
HYDRAULIC_INFO = {
    "name": "condition monitoring of hydraulic systems",
    "paper_ref": "[17] Helwig, Pignanelli, Schutze, I2MTC 2015",
    "url": "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip",
    "notes": (
        "The paper uses n=1449 rows taken under stable system settings, "
        "response = accumulator pressure (130/115/100/90), p=43680 from 17 sensors."
    ),
}

SP500_INFO = {
    "name": "S&P 500, year 2020",
    "paper_ref": "[18] FRED SP500; [19] Kaggle S&P 500 stocks",
    "fred_url": "https://fred.stlouisfed.org/series/SP500",
    "kaggle_url": "https://www.kaggle.com/hanseopark/sp-500-stocks-value-with-financial-statement",
    "notes": (
        "The paper uses 253 trading days in 2020. Response = S&P 500 index; "
        "predictors = constituent stock prices (some companies contribute more "
        "than one series)."
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path, *, timeout: int = 60) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as src, tmp.open("wb") as out:
            out.write(src.read())
        tmp.replace(dest)
    except Exception as exc:  # noqa: BLE001
        if tmp.exists():
            tmp.unlink()
        raise RuntimeError(
            f"Could not download {url}. Fetch the file by hand and place it at {dest}."
        ) from exc
    return dest


def fetch_hydraulic(
    cache_dir: Path | None = None,
    *,
    downloader: Callable[[str, Path], Path] | None = None,
) -> Path:
    """Download the UCI hydraulic-systems archive into ``cache_dir``.

    Returns the path to the zip. Parsing into the paper's ``(X, y)`` layout is
    left to ``reproduction/run_real_data.py``, because the 17-sensor expansion
    is study-specific.
    """
    root = cache_dir or DEFAULT_CACHE
    dest = root / "hydraulic" / "hydraulic_systems.zip"
    fetch = downloader or (lambda url, path: _download(url, path))
    return fetch(HYDRAULIC_INFO["url"], dest)


def fetch_sp500_info() -> dict[str, str]:
    """Return the documented S&P 500 source URLs. No automatic download.

    The Kaggle source requires an account, so this function only documents
    the locations. ``reproduction/run_real_data.py`` reads a user-supplied
    directory of CSVs.
    """
    return dict(SP500_INFO)


def verify_checksum(path: Path, expected: str) -> bool:
    """Compare a file to a published SHA-256 hex digest."""
    return _sha256(path) == expected.lower()
