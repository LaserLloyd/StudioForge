"""StudioForge companion: the ``sfctl`` CLI and its MCP bridge.

A deliberately thin client. All state -- the model registry, the VRAM planner,
the engine, the configuration -- lives on the StudioForge server and is reached
over HTTP, which is what lets this package be installed on any box (an OpenClaw
host, a laptop) without dragging the server's dependencies along.

Nothing here imports ``studioforge``.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
