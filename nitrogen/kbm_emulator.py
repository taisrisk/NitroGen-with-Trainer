from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import ctypes
from ctypes import wintypes


user32 = ctypes.WinDLL("user32", use_last_error=True)


# Some Python builds (including some venv/embeddable distributions) ship a reduced
# `ctypes.wintypes` without ULONG_PTR. Define a compatible pointer-sized unsigned type.
try:
    ULONG_PTR = wintypes.ULONG_PTR  # type: ignore[attr-defined]
except Exception:
    ULONG_PTR = ctypes.c_uint64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_uint32


INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_KEYUP = 0x0002

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_WHEEL = 0x0800
MOUSEEVENTF_XDOWN = 0x0080
MOUSEEVENTF_XUP = 0x0100

XBUTTON1 = 0x0001
XBUTTON2 = 0x0002
WHEEL_DELTA = 120


VK = {
    "shift": 0x10,
    "ctrl": 0x11,
    "alt": 0x12,
    "space": 0x20,
    "enter": 0x0D,
    "esc": 0x1B,
    "escape": 0x1B,
    "tab": 0x09,
    "caps_lock": 0x14,
    "backspace": 0x08,
    "delete": 0x2E,
    "insert": 0x2D,
    "home": 0x24,
    "end": 0x23,
    "page_up": 0x21,
    "page_down": 0x22,
    "arrow_left": 0x25,
    "arrow_up": 0x26,
    "arrow_right": 0x27,
    "arrow_down": 0x28,
    "`": 0xC0,  # VK_OEM_3
    "-": 0xBD,  # VK_OEM_MINUS
    "=": 0xBB,  # VK_OEM_PLUS
    "[": 0xDB,  # VK_OEM_4
    "]": 0xDD,  # VK_OEM_6
    "\\": 0xDC,  # VK_OEM_5
    ";": 0xBA,  # VK_OEM_1
    "'": 0xDE,  # VK_OEM_7
    ",": 0xBC,  # VK_OEM_COMMA
    ".": 0xBE,  # VK_OEM_PERIOD
    "/": 0xBF,  # VK_OEM_2
    "w": 0x57,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "b": 0x42,
    "h": 0x48,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "m": 0x4D,
    "n": 0x4E,
    "o": 0x4F,
    "p": 0x50,
    "t": 0x54,
    "u": 0x55,
    "y": 0x59,
    "e": 0x45,
    "q": 0x51,
    "r": 0x52,
    "f": 0x46,
    "g": 0x47,
    "c": 0x43,
    "v": 0x56,
    "x": 0x58,
    "z": 0x5A,
    "i": 0x49,
    "0": 0x30,
    "1": 0x31,
    "2": 0x32,
    "3": 0x33,
    "4": 0x34,
    "5": 0x35,
    "6": 0x36,
    "7": 0x37,
    "8": 0x38,
    "9": 0x39,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
}


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", _INPUT_UNION)]


user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
user32.SendInput.restype = wintypes.UINT


def _send_inputs(inputs: list[INPUT]) -> None:
    if not inputs:
        return
    n = user32.SendInput(len(inputs), (INPUT * len(inputs))(*inputs), ctypes.sizeof(INPUT))
    if n != len(inputs):
        err = ctypes.get_last_error()
        raise OSError(f"SendInput failed (sent {n}/{len(inputs)}): winerr={err}")


def _key_event(vk: int, down: bool) -> INPUT:
    flags = 0 if down else KEYEVENTF_KEYUP
    ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_KEYBOARD, union=_INPUT_UNION(ki=ki))


def _mouse_move(dx: int, dy: int) -> INPUT:
    mi = MOUSEINPUT(dx=dx, dy=dy, mouseData=0, dwFlags=MOUSEEVENTF_MOVE, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_MOUSE, union=_INPUT_UNION(mi=mi))


def _mouse_button(left: bool, down: bool) -> INPUT:
    if left:
        flags = MOUSEEVENTF_LEFTDOWN if down else MOUSEEVENTF_LEFTUP
    else:
        flags = MOUSEEVENTF_RIGHTDOWN if down else MOUSEEVENTF_RIGHTUP
    mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_MOUSE, union=_INPUT_UNION(mi=mi))


def _mouse_middle(down: bool) -> INPUT:
    flags = MOUSEEVENTF_MIDDLEDOWN if down else MOUSEEVENTF_MIDDLEUP
    mi = MOUSEINPUT(dx=0, dy=0, mouseData=0, dwFlags=flags, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_MOUSE, union=_INPUT_UNION(mi=mi))


def _mouse_xbutton(which: int, down: bool) -> INPUT:
    flags = MOUSEEVENTF_XDOWN if down else MOUSEEVENTF_XUP
    mouse_data = XBUTTON1 if int(which) == 1 else XBUTTON2
    mi = MOUSEINPUT(dx=0, dy=0, mouseData=wintypes.DWORD(int(mouse_data)), dwFlags=flags, time=0, dwExtraInfo=0)
    return INPUT(type=INPUT_MOUSE, union=_INPUT_UNION(mi=mi))


def _mouse_wheel(notches: int) -> INPUT:
    delta = int(notches) * int(WHEEL_DELTA)
    mi = MOUSEINPUT(
        dx=0,
        dy=0,
        mouseData=wintypes.DWORD(delta & 0xFFFFFFFF),
        dwFlags=MOUSEEVENTF_WHEEL,
        time=0,
        dwExtraInfo=0,
    )
    return INPUT(type=INPUT_MOUSE, union=_INPUT_UNION(mi=mi))


@dataclass
class KBMConfig:
    move_deadzone: float = 0.20
    mouse_deadzone: float = 0.01
    mouse_speed: float = 120.0  # pixels per step at stick=1.0
    trigger_threshold: float = 0.5
    require_foreground: bool = True


class KeyboardMouseEmulator:
    """
    Minimal user-mode KB/M injector for Windows using SendInput.

    - Uses a controller-token -> keys mapping (same JSON profile shape as training).
    - Converts left stick axes to WASD, right stick axes to mouse movement.
    - Converts button tokens to key presses / mouse clicks.
    """

    def __init__(
        self,
        *,
        token_to_keys: dict[str, list[str]],
        is_foreground: callable | None = None,
        config: KBMConfig | None = None,
    ):
        self.token_to_keys = {str(k).upper(): [str(x).lower() for x in (v or [])] for k, v in (token_to_keys or {}).items()}
        self.is_foreground = is_foreground
        self.cfg = config or KBMConfig()

        self._keys_down: set[str] = set()
        self._mouse_left_down = False
        self._mouse_right_down = False
        self._mouse_middle_down = False
        self._mouse_x1_down = False
        self._mouse_x2_down = False

    def release_all(self) -> None:
        inputs: list[INPUT] = []
        for k in sorted(self._keys_down):
            vk = VK.get(k)
            if vk is not None:
                inputs.append(_key_event(vk, down=False))
        if self._mouse_left_down:
            inputs.append(_mouse_button(left=True, down=False))
        if self._mouse_right_down:
            inputs.append(_mouse_button(left=False, down=False))
        if self._mouse_middle_down:
            inputs.append(_mouse_middle(down=False))
        if self._mouse_x1_down:
            inputs.append(_mouse_xbutton(1, down=False))
        if self._mouse_x2_down:
            inputs.append(_mouse_xbutton(2, down=False))
        _send_inputs(inputs)
        self._keys_down.clear()
        self._mouse_left_down = False
        self._mouse_right_down = False
        self._mouse_middle_down = False
        self._mouse_x1_down = False
        self._mouse_x2_down = False

    def _should_send(self) -> bool:
        if not self.cfg.require_foreground:
            return True
        if self.is_foreground is None:
            return True
        try:
            return bool(self.is_foreground())
        except Exception:
            return True

    def _set_key(self, key: str, down: bool, inputs: list[INPUT]) -> None:
        key = str(key).lower()
        if key in ("lmb", "rmb", "mmb", "x1", "x2", "scroll_up", "scroll_down", "mouse_x1", "mouse_x2"):
            return
        vk = VK.get(key)
        if vk is None:
            return
        if down:
            if key not in self._keys_down:
                inputs.append(_key_event(vk, down=True))
                self._keys_down.add(key)
        else:
            if key in self._keys_down:
                inputs.append(_key_event(vk, down=False))
                self._keys_down.remove(key)

    def _set_mouse_button(self, which: str, down: bool, inputs: list[INPUT]) -> None:
        which = str(which).lower()
        if which == "lmb":
            if down and not self._mouse_left_down:
                inputs.append(_mouse_button(left=True, down=True))
                self._mouse_left_down = True
            elif (not down) and self._mouse_left_down:
                inputs.append(_mouse_button(left=True, down=False))
                self._mouse_left_down = False
        elif which == "rmb":
            if down and not self._mouse_right_down:
                inputs.append(_mouse_button(left=False, down=True))
                self._mouse_right_down = True
            elif (not down) and self._mouse_right_down:
                inputs.append(_mouse_button(left=False, down=False))
                self._mouse_right_down = False
        elif which == "mmb":
            if down and not self._mouse_middle_down:
                inputs.append(_mouse_middle(down=True))
                self._mouse_middle_down = True
            elif (not down) and self._mouse_middle_down:
                inputs.append(_mouse_middle(down=False))
                self._mouse_middle_down = False
        elif which == "x1":
            if down and not self._mouse_x1_down:
                inputs.append(_mouse_xbutton(1, down=True))
                self._mouse_x1_down = True
            elif (not down) and self._mouse_x1_down:
                inputs.append(_mouse_xbutton(1, down=False))
                self._mouse_x1_down = False
        elif which == "x2":
            if down and not self._mouse_x2_down:
                inputs.append(_mouse_xbutton(2, down=True))
                self._mouse_x2_down = True
            elif (not down) and self._mouse_x2_down:
                inputs.append(_mouse_xbutton(2, down=False))
                self._mouse_x2_down = False

    def _axis_norm(self, v: int) -> float:
        v = max(-32768, min(32767, int(v)))
        return float(v) / 32767.0 if v >= 0 else float(v) / 32768.0

    def step(self, action: dict, *, duration_s: float) -> None:
        if not self._should_send():
            time.sleep(max(0.0, float(duration_s)))
            return

        desired_keys: set[str] = set()
        desired_lmb = False
        desired_rmb = False
        desired_mmb = False
        desired_x1 = False
        desired_x2 = False
        wheel_notches = 0

        # 1) Sticks -> WASD and mouse move.
        lx = 0.0
        ly = 0.0
        rx = 0.0
        ry = 0.0
        try:
            lx = self._axis_norm(action.get("AXIS_LEFTX", [0])[0])
            ly = self._axis_norm(action.get("AXIS_LEFTY", [0])[0])
            rx = self._axis_norm(action.get("AXIS_RIGHTX", [0])[0])
            ry = self._axis_norm(action.get("AXIS_RIGHTY", [0])[0])
        except Exception:
            pass

        # This codebase uses XInput-style axes:
        # - Left stick Y: negative = up/forward, positive = down/back.
        dz = float(self.cfg.move_deadzone)
        if ly < -dz:
            desired_keys.add("w")
        elif ly > dz:
            desired_keys.add("s")
        if lx > dz:
            desired_keys.add("d")
        elif lx < -dz:
            desired_keys.add("a")

        mdz = float(self.cfg.mouse_deadzone)
        if abs(rx) >= mdz or abs(ry) >= mdz:
            dx = int(rx * float(self.cfg.mouse_speed))
            dy = int(ry * float(self.cfg.mouse_speed))
        else:
            dx = 0
            dy = 0

        # 2) Controller tokens -> keys/mouse buttons via profile.
        for token, keys in self.token_to_keys.items():
            if not keys:
                continue
            v = action.get(token)
            pressed = False
            if isinstance(v, (int, bool)):
                pressed = bool(v)
            else:
                try:
                    if isinstance(v, (list, tuple)) and v:
                        pressed = float(v[0]) > 0.5
                    elif hasattr(v, "shape"):
                        pressed = float(v[0]) > 0.5
                except Exception:
                    pressed = False

            # Trigger arrays are 0..255 in this codebase.
            if "TRIGGER" in token:
                try:
                    if isinstance(v, (list, tuple)) and v:
                        pressed = (float(v[0]) / 255.0) >= float(self.cfg.trigger_threshold)
                except Exception:
                    pass

            if not pressed:
                continue
            for k in keys:
                k = str(k).lower()
                if k == "lmb":
                    desired_lmb = True
                elif k == "rmb":
                    desired_rmb = True
                elif k in ("mmb", "middle_click"):
                    desired_mmb = True
                elif k in ("x1", "mouse_x1"):
                    desired_x1 = True
                elif k in ("x2", "mouse_x2"):
                    desired_x2 = True
                elif k == "scroll_up":
                    wheel_notches += 1
                elif k == "scroll_down":
                    wheel_notches -= 1
                else:
                    desired_keys.add(k)

        # 3) Emit diffs.
        inputs: list[INPUT] = []
        for k in sorted(set(self._keys_down) - desired_keys):
            self._set_key(k, down=False, inputs=inputs)
        for k in sorted(desired_keys - set(self._keys_down)):
            self._set_key(k, down=True, inputs=inputs)

        self._set_mouse_button("lmb", desired_lmb, inputs)
        self._set_mouse_button("rmb", desired_rmb, inputs)
        self._set_mouse_button("mmb", desired_mmb, inputs)
        self._set_mouse_button("x1", desired_x1, inputs)
        self._set_mouse_button("x2", desired_x2, inputs)
        if dx or dy:
            inputs.append(_mouse_move(dx, dy))
        if wheel_notches:
            inputs.append(_mouse_wheel(int(wheel_notches)))

        if inputs:
            _send_inputs(inputs)

        time.sleep(max(0.0, float(duration_s)))

    def step_keys(
        self,
        *,
        keys_down: set[str] | Iterable[str],
        lmb: bool = False,
        rmb: bool = False,
        mmb: bool = False,
        x1: bool = False,
        x2: bool = False,
        wheel: int = 0,
        mouse_dx: int = 0,
        mouse_dy: int = 0,
        duration_s: float,
    ) -> None:
        """
        Apply a "pure keyboard/mouse" action:
        - keys_down: iterable of key names (e.g. 'w', 'shift', 'e')
        - lmb/rmb: mouse button state
        - mouse_dx/mouse_dy: relative mouse move in pixels for this step
        """
        if not self._should_send():
            time.sleep(max(0.0, float(duration_s)))
            return

        desired_keys = {str(k).lower().strip() for k in keys_down if str(k).strip()}
        inputs: list[INPUT] = []
        for k in sorted(set(self._keys_down) - desired_keys):
            self._set_key(k, down=False, inputs=inputs)
        for k in sorted(desired_keys - set(self._keys_down)):
            self._set_key(k, down=True, inputs=inputs)

        self._set_mouse_button("lmb", bool(lmb), inputs)
        self._set_mouse_button("rmb", bool(rmb), inputs)
        self._set_mouse_button("mmb", bool(mmb), inputs)
        self._set_mouse_button("x1", bool(x1), inputs)
        self._set_mouse_button("x2", bool(x2), inputs)
        dx = int(mouse_dx or 0)
        dy = int(mouse_dy or 0)
        if dx or dy:
            inputs.append(_mouse_move(dx, dy))
        if int(wheel or 0):
            inputs.append(_mouse_wheel(int(wheel)))

        if inputs:
            _send_inputs(inputs)

        time.sleep(max(0.0, float(duration_s)))
