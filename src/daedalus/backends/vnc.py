"""VNC backend wrapping ``vncdotool``'s synchronous API.

This is the production backend for Phase 0/1. It targets any host running a
VNC server (Windows TightVNC/UltraVNC, macOS Screen Sharing, Linux TigerVNC).

vncdotool's sync API uses a Twisted reactor on a background thread and exposes
a blocking client. We adapt it to our :class:`RemoteDesktop` protocol.

Notes
-----
- ``captureScreen(path)`` writes a PNG to disk, so we use a per-call tempfile.
  This is the conservative path; future optimisation can read framebuffer
  pixels directly via the underlying ``VNCDoToolClient.screen`` attribute.
- Key names mostly follow X11 keysym conventions, which vncdotool already
  understands. We translate a small set of common names.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
import time
from pathlib import Path

from PIL import Image

from daedalus.backends.protocol import Button, Rect, Screenshot
from daedalus.core.errors import BackendError

log = logging.getLogger(__name__)


# Map our friendly key names to vncdotool/X11 keysyms. Anything not in the
# table is passed through verbatim, which covers single ASCII characters,
# ``f1``..``f12``, and named keys like ``enter``, ``tab``, ``escape``.
_KEY_ALIASES = {
    # Modifiers
    "ctrl": "ctrl",
    "control": "ctrl",
    "shift": "shift",
    "alt": "alt",
    "option": "alt",
    "opt": "alt",
    "meta": "super",
    "super": "super",
    "win": "super",
    "cmd": "super",
    "command": "super",
    # Navigation
    "esc": "esc",
    "escape": "esc",
    "return": "enter",
    "enter": "enter",
    "del": "del",
    "delete": "delete",
    "bksp": "bsp",
    "backspace": "bsp",
    "pgup": "pgup",
    "pageup": "pgup",
    "pgdn": "pgdn",
    "pagedown": "pgdn",
    "home": "home",
    "end": "end",
    "insert": "ins",
    "ins": "ins",
    # Arrow keys
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "arrowleft": "left",
    "arrowright": "right",
    "arrowup": "up",
    "arrowdown": "down",
    # Whitespace
    "space": "space",
    "spacebar": "space",
    "tab": "tab",
    # Special
    "capslock": "caplk",
    "numlock": "numlk",
    "scrolllock": "scrlk",
    "pause": "pause",
    "printscreen": "sysrq",
    # Function keys — vncdotool uses lowercase "fN" keysym names
    "f1": "f1",
    "f2": "f2",
    "f3": "f3",
    "f4": "f4",
    "f5": "f5",
    "f6": "f6",
    "f7": "f7",
    "f8": "f8",
    "f9": "f9",
    "f10": "f10",
    "f11": "f11",
    "f12": "f12",
}

_BUTTON_TO_INT = {Button.LEFT: 1, Button.MIDDLE: 2, Button.RIGHT: 3}


class VNCBackend:
    """Synchronous VNC backend.

    Parameters
    ----------
    host:
        Hostname or IP of the VNC server.
    port:
        TCP port. Default 5900.
    password:
        Plain-text password. Pass ``None`` for servers without authentication.
        Prefer reading from an env var rather than passing in source.
    timeout_s:
        Per-call timeout the underlying vncdotool client will enforce.
    max_resolution:
        Optional ``(width, height)`` cap. When set, screenshots that exceed
        this size are downscaled (preserving aspect ratio) and all coordinate
        inputs/outputs are mapped between the logical and native spaces. This
        is essential for macOS Retina displays where the VNC server exposes
        the native pixel resolution even when the user has chosen a lower
        "looks like" scaling in System Settings.
    """

    def __init__(
        self,
        host: str,
        port: int = 5900,
        password: str | None = None,
        username: str | None = None,
        timeout_s: float = 10.0,
        max_resolution: tuple[int, int] | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._password = password
        self._username = username
        self._timeout_s = timeout_s
        self._max_resolution = max_resolution
        self._client = None  # type: ignore[var-annotated]
        self._native_size: tuple[int, int] = (0, 0)
        self._scale: float = 1.0  # logical-to-native multiplier

    @property
    def size(self) -> tuple[int, int]:
        if self._max_resolution and self._scale != 1.0:
            mw, mh = self._max_resolution
            nw, nh = self._native_size
            return (min(nw, mw), min(nh, mh))
        return self._native_size

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # -- Lifecycle -----------------------------------------------------------------

    def connect(self) -> None:
        try:
            from vncdotool import api  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional at install time
            raise BackendError("vncdotool is not installed") from exc

        # vncdotool address format: "host::port" (two colons) or "host:display"
        addr = f"{self._host}::{self._port}"
        try:
            self._client = api.connect(addr, password=self._password, username=self._username)
        except Exception as exc:
            raise BackendError(f"VNC connect failed for {addr}: {exc}") from exc
        if self._client is None:  # pragma: no cover
            raise BackendError("vncdotool returned a null client")
        with contextlib.suppress(Exception):
            self._client.timeout = self._timeout_s

        # Force a full framebuffer update from the server. Without this,
        # many VNC servers (notably TightVNC on Windows) won't push pixel
        # data until the client explicitly requests it, resulting in an
        # all-black capture on the first screenshot.
        with contextlib.suppress(Exception):
            self._client.refreshScreen()

        # Resolve real screen size by capturing one frame.
        try:
            shot = self._raw_screenshot()
            self._native_size = (shot.width, shot.height)
        except Exception as exc:
            log.warning("could not auto-detect VNC screen size: %s", exc)

        # Compute downscale factor for Retina / HiDPI displays.
        if self._max_resolution:
            mw, mh = self._max_resolution
            nw, nh = self._native_size
            if nw > mw or nh > mh:
                self._scale = max(nw / mw, nh / mh)
                logical = (round(nw / self._scale), round(nh / self._scale))
                log.info(
                    "VNC native %dx%d -> logical %dx%d (scale=%.3f)",
                    nw, nh, logical[0], logical[1], self._scale,
                )
            else:
                self._scale = 1.0

    def disconnect(self) -> None:
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception as exc:  # pragma: no cover
                log.warning("VNC disconnect: %s", exc)
            self._client = None

    def close(self) -> None:
        """Disconnect and stop the Twisted reactor. Call only when completely done."""
        self.disconnect()
        try:
            from vncdotool import api
            api.shutdown()
        except Exception as exc:  # pragma: no cover
            log.warning("vncdotool reactor shutdown: %s", exc)

    # -- Capture -------------------------------------------------------------------

    def _raw_screenshot(self, region: Rect | None = None) -> Screenshot:
        """Capture at native VNC resolution (no downscaling)."""
        client = self._require_client("screenshot")
        with contextlib.suppress(Exception):
            client.refreshScreen()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            try:
                if region is not None:
                    client.captureRegion(
                        str(tmp_path), region.x, region.y, region.width, region.height
                    )
                else:
                    client.captureScreen(str(tmp_path))
            except Exception as exc:
                raise BackendError(f"VNC screen capture failed: {exc}") from exc
            img = Image.open(tmp_path).convert("RGBA")
            return Screenshot(
                image=img,
                width=img.width,
                height=img.height,
                captured_at=time.time(),
            )
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

    def screenshot(self, region: Rect | None = None) -> Screenshot:
        # When downscaling, translate logical region coords to native coords.
        native_region = region
        if region is not None and self._scale != 1.0:
            native_region = Rect(
                x=round(region.x * self._scale),
                y=round(region.y * self._scale),
                width=round(region.width * self._scale),
                height=round(region.height * self._scale),
            )
        try:
            shot = self._raw_screenshot(native_region)
        except BackendError:
            self._reconnect()
            shot = self._raw_screenshot(native_region)

        if self._scale != 1.0:
            new_w = round(shot.width / self._scale)
            new_h = round(shot.height / self._scale)
            img = shot.image.resize((new_w, new_h), Image.LANCZOS)
            return Screenshot(
                image=img,
                width=new_w,
                height=new_h,
                captured_at=shot.captured_at,
            )
        return shot

    # -- Mouse ---------------------------------------------------------------------

    def move(self, x: int, y: int) -> None:
        client = self._require_client("move")
        nx, ny = round(x * self._scale), round(y * self._scale)
        try:
            client.mouseMove(nx, ny)
        except Exception as exc:
            raise BackendError(f"VNC mouseMove failed: {exc}") from exc

    def click(
        self,
        x: int,
        y: int,
        button: Button = Button.LEFT,
        double: bool = False,
    ) -> None:
        client = self._require_client("click")
        btn = _BUTTON_TO_INT[button]
        nx, ny = round(x * self._scale), round(y * self._scale)
        try:
            client.mouseMove(nx, ny)
            client.mousePress(btn)
            if double:
                time.sleep(0.06)
                client.mousePress(btn)
        except Exception as exc:
            raise BackendError(f"VNC click failed: {exc}") from exc

    def drag(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        button: Button = Button.LEFT,
    ) -> None:
        client = self._require_client("drag")
        btn = _BUTTON_TO_INT[button]
        nx1, ny1 = round(x1 * self._scale), round(y1 * self._scale)
        nx2, ny2 = round(x2 * self._scale), round(y2 * self._scale)
        try:
            client.mouseMove(nx1, ny1)
            client.mouseDown(btn)
            client.mouseMove(nx2, ny2)
            client.mouseUp(btn)
        except Exception as exc:
            raise BackendError(f"VNC drag failed: {exc}") from exc

    def scroll(self, dx: int, dy: int) -> None:
        client = self._require_client("scroll")
        # vncdotool exposes scroll as buttons 4 (up) and 5 (down).
        try:
            for _ in range(abs(dy)):
                client.mousePress(4 if dy < 0 else 5)
            for _ in range(abs(dx)):
                client.mousePress(6 if dx < 0 else 7)
        except Exception as exc:
            raise BackendError(f"VNC scroll failed: {exc}") from exc

    # -- Keyboard ------------------------------------------------------------------

    def write(self, text: str) -> None:
        client = self._require_client("write")
        try:
            for ch in text:
                if ch == "\n":
                    client.keyPress("enter")
                elif ch == "\t":
                    client.keyPress("tab")
                elif ch.isupper() and ch.isascii():
                    client.keyDown("shift")
                    client.keyPress(ch.lower())
                    client.keyUp("shift")
                else:
                    client.keyPress(ch)
        except Exception as exc:
            raise BackendError(f"VNC write failed: {exc}") from exc

    def press(self, *keys: str) -> None:
        client = self._require_client("press")
        if not keys:
            return
        translated = [self._translate_key(k) for k in keys]
        try:
            self._do_press(client, translated)
        except BackendError:
            self._reconnect()
            client = self._require_client("press")
            self._do_press(client, translated)

    def _do_press(self, client, translated: list[str]) -> None:  # type: ignore[no-untyped-def]
        try:
            if len(translated) == 1:
                client.keyPress(translated[0])
                return
            # Modifier combo: hold all but last, press last, release in reverse.
            holds = translated[:-1]
            tap = translated[-1]
            for k in holds:
                client.keyDown(k)
            try:
                client.keyPress(tap)
            finally:
                for k in reversed(holds):
                    client.keyUp(k)
        except Exception as exc:
            raise BackendError(f"VNC press failed: {exc}") from exc

    # -- Internal ------------------------------------------------------------------

    def _require_client(self, op: str):  # type: ignore[no-untyped-def]
        if self._client is None:
            raise BackendError(f"VNC backend not connected (op={op})")
        return self._client

    def _reconnect(self) -> None:
        """Attempt to reconnect after a connection failure."""
        log.warning("VNC connection appears broken, attempting reconnect...")
        try:
            self.disconnect()
        except Exception:
            self._client = None
        try:
            self.connect()
            log.info("VNC reconnect successful")
        except Exception as exc:
            raise BackendError(f"VNC reconnect failed: {exc}") from exc

    @staticmethod
    def _translate_key(name: str) -> str:
        lower = name.lower()
        return _KEY_ALIASES.get(lower, lower)
