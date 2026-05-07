"""Backend-agnostic remote desktop interface.

Skills depend only on this file. Concrete backends (VNC, mock, future RDP /
local / browser) implement the :class:`RemoteDesktop` Protocol.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


class Button(enum.StrEnum):
    LEFT = "left"
    MIDDLE = "middle"
    RIGHT = "right"


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int


@dataclass
class Screenshot:
    """A captured frame. ``image`` is RGBA. ``captured_at`` is unix epoch seconds."""

    image: Image.Image
    width: int
    height: int
    captured_at: float


@runtime_checkable
class RemoteDesktop(Protocol):
    """The narrow surface every backend exposes to skills.

    All methods are synchronous; backends that are async internally (e.g. the
    Twisted-based VNC client) wrap their async API behind a sync facade.
    """

    @property
    def size(self) -> tuple[int, int]: ...

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def close(self) -> None:
        """Disconnect and release all resources (e.g. background reactor threads).
        Default implementation delegates to ``disconnect()``."""
        self.disconnect()

    def screenshot(self, region: Rect | None = None) -> Screenshot: ...

    def move(self, x: int, y: int) -> None: ...

    def click(
        self,
        x: int,
        y: int,
        button: Button = Button.LEFT,
        double: bool = False,
    ) -> None: ...

    def write(self, text: str) -> None:
        """Type a literal text string. Backends that don't natively support
        unicode should iterate per-character via :meth:`press`."""

    def press(self, *keys: str) -> None:
        """Press one or more keys simultaneously and release them.

        ``keys`` is a sequence of key names. Single-character keys are the
        character itself (``"a"``, ``"A"``, ``"1"``). Modifiers and named keys
        use the X11 keysym lowercase form: ``"ctrl"``, ``"shift"``, ``"alt"``,
        ``"super"``, ``"enter"``, ``"esc"``, ``"tab"``, ``"backspace"``,
        ``"delete"``, ``"home"``, ``"end"``, ``"up"``, ``"down"``, ``"left"``,
        ``"right"``, ``"f1"``..``"f12"``.

        Examples::

            press("enter")
            press("ctrl", "c")
            press("ctrl", "shift", "t")
        """

    def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        button: Button = Button.LEFT,
    ) -> None:
        """Press at (x1,y1), drag to (x2,y2), and release."""

    def mouse_down(self, button: Button = Button.LEFT) -> None:
        """Press and hold a mouse button at the current cursor position."""

    def mouse_up(self, button: Button = Button.LEFT) -> None:
        """Release a mouse button at the current cursor position."""

    def scroll(self, dx: int, dy: int) -> None: ...
