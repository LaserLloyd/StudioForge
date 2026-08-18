"""Windows system-tray front end for StudioForge.

Nothing is imported eagerly here: :mod:`studioforge.tray.tray_app` pulls in
``pystray``, which is an optional extra on a headless install. Importing this
package must stay free so ``studioforge tray`` can report a missing dependency
as a sentence rather than an ``ImportError`` traceback.
"""

from __future__ import annotations

from studioforge.config import Config

#: Shown verbatim when ``pystray`` (or its backend) cannot be imported.
MISSING_PYSTRAY_HINT = (
    "the system tray needs 'pystray', which is not installed in this environment.\n"
    "  install it with:  pip install pystray\n"
    "  (Pillow, the other requirement, is already a StudioForge dependency)"
)

__all__ = ["MISSING_PYSTRAY_HINT", "main"]


def main(config: Config | None = None) -> int:
    """Run the tray. Imported lazily so ``pystray`` is only required on use."""
    from studioforge.tray.tray_app import main as _main

    return _main(config)
