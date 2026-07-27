"""Windows-safe console output helpers."""
from __future__ import annotations

import sys


def configure_stdout() -> None:
    """Use UTF-8 on the console when possible (avoids cp1254 UnicodeEncodeError)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass


def safe_print_df(df, columns) -> None:
    """Print a DataFrame without Unicode symbols that break Turkish Windows consoles."""
    aliases = {
        "PM2.5 (μg/m³)": "PM2.5 (ug/m3)",
        "NO₂ (μg/m³)": "NO2 (ug/m3)",
    }
    view = df[columns].rename(columns=aliases)
    print(view.to_string(index=False))
