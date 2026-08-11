"""Console setup shared by the entry-point scripts.

Progress lines carry check/cross marks and party names in 21 languages. On Windows the
default stdout encoding is the ANSI code page, so a redirected run (`> log.txt`, or the
pipes `analyze_all.py` opens around each child) dies with UnicodeEncodeError on the first
non-Latin-1 character. Every script used to paste the same `sys.stdout.reconfigure(...)`
line near its imports; it lives here instead.
"""
import sys


def configure_stdout() -> None:
    """Force UTF-8 on stdout/stderr, replacing characters the terminal cannot render."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
