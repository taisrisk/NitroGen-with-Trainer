import os
import sys
import time
import json
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
from nitrogen.shared import BUTTON_ACTION_TOKENS, PATH_REPO
from nitrogen.inference_viz import create_viz, VideoRecorder
from nitrogen.inference_client import ModelClient
from nitrogen.kbm_emulator import KeyboardMouseEmulator, KBMConfig
from ollama_system2 import OllamaSystem2

try:
    import win32api
    import win32con
    import win32gui
except Exception:
    win32api = None
    win32con = None
    win32gui = None

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
parser.add_argument("--input-backend", choices=["controller", "kbm"], default="controller", help="How to execute actions: virtual controller (ViGEm) or direct keyboard/mouse")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--timeout-ms", type=int, default=120_000, help="Model server receive timeout (ms)")
parser.add_argument("--wait-for-server-s", type=int, nargs="?", const=60, default=60, help="Wait up to N seconds for the model server to respond (0 disables)")
parser.add_argument("--focus-each-step", action="store_true", help="Refocus the game window each step (helps some games accept virtual controller)")
parser.add_argument("--debug", action="store_true", help="Print every sub-action sent to the controller")
parser.add_argument("--action-hold-ms", type=int, default=50, help="Hold each sub-action for at least this many ms (improves input reliability)")
parser.add_argument("--controller-keepalive-every", type=int, default=0, help="Every N env steps, reset controller state (0 disables)")
parser.add_argument("--allow-user-override", action="store_true", help="Hold a key to temporarily disable AI inputs (keyboard/mouse play along)")
parser.add_argument("--override-key", choices=["shift", "ctrl", "alt"], default="shift", help="Key used for --allow-user-override")
parser.add_argument("--button-threshold", type=float, default=0.5, help="Button press threshold (lower presses more)")
parser.add_argument("--stick-deadzone", type=float, default=0.08, help="Joystick deadzone in normalized units (applies to left stick; right stick defaults lower)")
parser.add_argument("--right-stick-deadzone", type=float, default=0.02, help="Right-stick deadzone (camera). Lower = more camera motion, higher = less drift")
parser.add_argument("--stick-gain", type=float, default=1.0, help="Multiply joystick magnitude before clipping")
parser.add_argument("--right-stick-gain", type=float, default=None, help="Optional right-stick gain override (defaults to --stick-gain)")
parser.add_argument("--controller-type", choices=["xbox", "ps4"], default="xbox", help="Virtual controller type")
parser.add_argument("--force-controller-pulse-on-start", action="store_true", help="Pulse a button at startup to force controller prompts/mode")
parser.add_argument("--force-controller-pulse-every", type=int, default=0, help="Every N model steps, pulse a button to keep controller mode (0 disables)")
parser.add_argument("--force-controller-button", type=str, default="SOUTH", help="Button name to pulse (e.g. SOUTH, EAST, DPAD_UP)")
parser.add_argument("--force-controller-pulse-ms", type=int, default=120, help="Pulse duration (ms)")
parser.add_argument("--env-fps", type=int, default=30, help="Gamepad action FPS (lower improves capture stability)")
parser.add_argument("--obs-width", type=int, default=1280, help="Captured observation width (resized)")
parser.add_argument("--obs-height", type=int, default=720, help="Captured observation height (resized)")
parser.add_argument("--screenshot-backend", choices=["dxcam", "pyautogui"], default="dxcam", help="Screenshot backend")
parser.add_argument("--capture-margin", type=int, default=0, help="Crop margin (pixels) from each window edge before capture")
parser.add_argument("--reuse-last-frame-on-fail", action=argparse.BooleanOptionalAction, default=True, help="If capture fails, reuse the previous frame instead of waiting")
parser.add_argument("--capture-retry-delay-ms", type=int, default=20, help="Delay between capture retries (ms)")
parser.add_argument("--capture-hard-fail-after", type=int, default=5, help="After N consecutive capture fails, pause actions more aggressively")
parser.add_argument("--capture-hard-fail-sleep-ms", type=int, default=1000, help="Extra sleep after hard-fail threshold (ms)")
parser.add_argument("--test-controller", action="store_true", help="Interactive controller sanity test before running the model loop")
parser.add_argument("--trace-model", action="store_true", help="Print model outputs (raw buttons/triggers + stick values) each step")
parser.add_argument("--trace-model-topk", type=int, default=8, help="How many top button scores to print with --trace-model")
parser.add_argument(
    "--invert-left-y",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Invert model left-stick Y before execution (default: True; fixes forward/back + menu scrolling).",
)
parser.add_argument(
    "--invert-right-y",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Invert model right-stick Y before execution (default: True; fixes camera look Y).",
)
parser.add_argument("--disable-dpad", action="store_true", help="Force DPAD_* = 0 (useful if the model spams menus/weapon select)")
parser.add_argument(
    "--emulator-invert-y",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Invert Y axes inside the virtual controller driver layer (use if forward/back is flipped).",
)
parser.add_argument("--profile", type=str, default="aio", help="Profile name (profiles/<name>.json) or path to JSON (used for --input-backend kbm token->key mapping)")
parser.add_argument("--kbm-mouse-speed", type=float, default=120.0, help="Mouse pixels-per-step at stick=1.0 for --input-backend kbm")
parser.add_argument("--kbm-move-deadzone", type=float, default=0.20, help="Left-stick deadzone for WASD mapping in --input-backend kbm")
parser.add_argument("--kbm-mouse-deadzone", type=float, default=0.01, help="Right-stick deadzone for mouse mapping in --input-backend kbm")
parser.add_argument("--kbm-trigger-threshold", type=float, default=0.5, help="Trigger threshold (0..1) for token->key mapping in --input-backend kbm")
parser.add_argument("--kbm-require-foreground", action=argparse.BooleanOptionalAction, default=True, help="Only inject KB/M when the game window is foreground (recommended)")

args = parser.parse_args()

if args.emulator_invert_y and (args.invert_left_y or args.invert_right_y):
    print(
        "[warn] Both --emulator-invert-y and model-side inversion are enabled; this may double-invert Y axes. "
        "Prefer leaving --emulator-invert-y off."
    )

policy = ModelClient(port=args.port, timeout_ms=args.timeout_ms)
if args.wait_for_server_s and args.wait_for_server_s > 0:
    policy.wait_for_server(timeout_s=float(args.wait_for_server_s))

for attempt in range(1, 6):
    try:
        policy.reset()
        policy_info = policy.info()
        break
    except Exception as e:
        if attempt == 5:
            raise
        print(f"Model server not ready ({e!r}); reconnecting...")
        policy.reconnect()
        time.sleep(0.5 * attempt)
action_downsample_ratio = policy_info["action_downsample_ratio"]
server_action_space = policy_info.get("action_space")
server_button_names = policy_info.get("button_names")
if str(server_action_space or "").strip().lower() != "keyboard":
    raise SystemExit(
        f"Controller/25D action_space is no longer supported. Server reported action_space={server_action_space!r}. "
        "Re-train a keyboard action-space model and serve that checkpoint."
    )
IS_KEYBOARD_MODEL = True
if not isinstance(server_button_names, list) or not server_button_names:
    raise SystemExit("Server reports keyboard action_space but did not provide button_names in info().")
TOKEN_SET = [str(x).strip().lower() for x in server_button_names if str(x).strip()]
if args.input_backend != "kbm":
    raise SystemExit("Keyboard action_space requires `--input-backend kbm`.")
TOKEN_SET_LOWER = set(TOKEN_SET)
expected_command_id: int | None = None
last_cmd_id: int | None = None

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = float(args.button_threshold)

# Find in path_out the list of existing video files, named 0001.mp4, 0002.mp4, etc.
# If they exist, find the max number and set the next number to be max + 1
video_files = sorted(PATH_OUT.glob("*_DEBUG.mp4"))
if video_files:
    existing_numbers = [f.name.split("_")[0] for f in video_files]
    existing_numbers = [int(n) for n in existing_numbers if n.isdigit()]
    next_number = max(existing_numbers) + 1
else:
    next_number = 1

PATH_MP4_DEBUG = PATH_OUT / f"{next_number:04d}_DEBUG.mp4"
PATH_MP4_CLEAN = PATH_OUT / f"{next_number:04d}_CLEAN.mp4"
PATH_ACTIONS = PATH_OUT / f"{next_number:04d}_ACTIONS.json"

def preprocess_img(main_image):
    main_cv = cv2.cvtColor(np.array(main_image), cv2.COLOR_RGB2BGR)
    final_image = cv2.resize(main_cv, (256, 256), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

zero_action = OrderedDict(
        [ 
            ("WEST", 0),
            ("SOUTH", 0),
            ("BACK", 0),
            ("DPAD_DOWN", 0),
            ("DPAD_LEFT", 0),
            ("DPAD_RIGHT", 0),
            ("DPAD_UP", 0),
            ("GUIDE", 0),
            ("AXIS_LEFTX", np.array([0], dtype=np.int32)),
            ("AXIS_LEFTY", np.array([0], dtype=np.int32)),
            ("LEFT_SHOULDER", 0),
            ("LEFT_TRIGGER", np.array([0], dtype=np.int32)),
            ("AXIS_RIGHTX", np.array([0], dtype=np.int32)),
            ("AXIS_RIGHTY", np.array([0], dtype=np.int32)),
            ("LEFT_THUMB", 0),
            ("RIGHT_THUMB", 0),
            ("RIGHT_SHOULDER", 0),
            ("RIGHT_TRIGGER", np.array([0], dtype=np.int32)),
            ("START", 0),
            ("EAST", 0),
            ("NORTH", 0),
        ]
    )


def _override_pressed() -> bool:
    if not args.allow_user_override:
        return False
    if win32api is None or win32con is None:
        return False

    vk = {
        "shift": win32con.VK_SHIFT,
        "ctrl": win32con.VK_CONTROL,
        "alt": win32con.VK_MENU,
    }[args.override_key]
    return bool(win32api.GetAsyncKeyState(vk) & 0x8000)


def _format_sub_action(idx: int, move_action: OrderedDict, token_set, *, cmd_id: int | None = None) -> str:
    lx = int(move_action["AXIS_LEFTX"][0])
    ly = int(move_action["AXIS_LEFTY"][0])
    rx = int(move_action["AXIS_RIGHTX"][0])
    ry = int(move_action["AXIS_RIGHTY"][0])
    lt = int(move_action.get("LEFT_TRIGGER", [0])[0])
    rt = int(move_action.get("RIGHT_TRIGGER", [0])[0])

    pressed = []
    for name in token_set:
        if "TRIGGER" in name:
            continue
        v = move_action.get(name, 0)
        if isinstance(v, np.ndarray):
            v = int(v[0])
        if int(v) != 0:
            pressed.append(name)

    buttons_s = ",".join(pressed) if pressed else "-"
    prefix = f"[cmd {cmd_id}] " if isinstance(cmd_id, int) else ""
    return f"{prefix}[{idx:02d}] LX={lx:6d} LY={ly:6d} RX={rx:6d} RY={ry:6d} LT={lt:3d} RT={rt:3d} btn={buttons_s}"


def _format_model_trace(
    idx: int,
    j_left_i: np.ndarray,
    j_right_i: np.ndarray,
    buttons_raw_i: np.ndarray | None,
    token_set: list[str],
    topk: int,
) -> str:
    jl_x, jl_y = float(j_left_i[0]), float(j_left_i[1])
    jr_x, jr_y = float(j_right_i[0]), float(j_right_i[1])
    jl_mag = float(np.hypot(jl_x, jl_y))
    jr_mag = float(np.hypot(jr_x, jr_y))

    parts = [
        f"[model {idx:02d}]",
        f"jL=({jl_x:+.3f},{jl_y:+.3f})|{jl_mag:.3f}",
        f"jR=({jr_x:+.3f},{jr_y:+.3f})|{jr_mag:.3f}",
    ]

    if buttons_raw_i is None:
        parts.append("buttons_raw=None")
        return " ".join(parts)

    topk = max(0, int(topk))
    if topk == 0:
        return " ".join(parts)

    # Sort by descending "activation" (use raw value; triggers are also in [0,1] typically).
    pairs = list(zip(token_set, [float(x) for x in buttons_raw_i]))
    pairs.sort(key=lambda kv: kv[1], reverse=True)
    shown = pairs[:topk]
    tops = ", ".join([f"{k}={v:.3f}" for k, v in shown])
    parts.append(f"top[{topk}]: {tops}")
    return " ".join(parts)

print("Model loaded, starting environment...")
for i in range(3):
    print(f"{3 - i}...")
    time.sleep(1)

env = GamepadEnv(
    game=args.process,
    game_speed=1.0,
    env_fps=args.env_fps,
    async_mode=True,
    focus_each_step=args.focus_each_step,
    image_width=args.obs_width,
    image_height=args.obs_height,
    controller_type=args.controller_type,
    enable_controller=(args.input_backend == "controller"),
    screenshot_backend=args.screenshot_backend,
    capture_margin=args.capture_margin,
    reuse_last_frame_on_capture_fail=args.reuse_last_frame_on_fail,
    capture_retry_delay_s=args.capture_retry_delay_ms / 1000.0,
    hard_fail_after=args.capture_hard_fail_after,
    hard_fail_sleep_s=args.capture_hard_fail_sleep_ms / 1000.0,
    invert_y_axis=bool(args.emulator_invert_y),
)


def _resolve_profile_path(profile: str) -> Path:
    s = str(profile or "").strip()
    if not s:
        raise ValueError("Empty --profile")
    p = Path(s)
    if p.suffix.lower() == ".json":
        if p.exists():
            return p.resolve()
        # Allow `--profile gow.json` to refer to profiles/gow.json
        return (PATH_REPO / "profiles" / p).resolve()
    return (PATH_REPO / "profiles" / f"{p.name}.json").resolve()


def _load_profile_bindings(profile: str) -> dict[str, list[str]]:
    path = _resolve_profile_path(profile)
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict) or not isinstance(obj.get("bindings"), dict):
        raise ValueError(f"Invalid profile JSON (missing bindings): {path}")
    bindings: dict[str, list[str]] = {}
    for token, keys in obj["bindings"].items():
        token_s = str(token).strip().upper()
        if isinstance(keys, str):
            keys_list = [keys]
        elif isinstance(keys, (list, tuple)):
            keys_list = list(keys)
        else:
            continue
        bindings[token_s] = [str(k).strip().lower() for k in keys_list if str(k).strip()]
    return bindings


def _is_game_foreground() -> bool:
    if win32gui is None:
        return True
    hwnd = getattr(env, "game_hwnd", None)
    if hwnd is None:
        return True
    try:
        return int(win32gui.GetForegroundWindow()) == int(hwnd)
    except Exception:
        return True


kbm = None
if args.input_backend == "kbm":
    token_to_keys = {} if IS_KEYBOARD_MODEL else _load_profile_bindings(args.profile)
    kbm = KeyboardMouseEmulator(
        token_to_keys=token_to_keys,
        is_foreground=_is_game_foreground,
        config=KBMConfig(
            move_deadzone=float(args.kbm_move_deadzone),
            mouse_deadzone=float(args.kbm_mouse_deadzone),
            mouse_speed=float(args.kbm_mouse_speed),
            trigger_threshold=float(args.kbm_trigger_threshold),
            require_foreground=bool(args.kbm_require_foreground),
        ),
    )

# These games requires to open a menu to initialize the controller
if args.process == "isaac-ng.exe" and env.gamepad_emulator is not None:
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

if args.process == "Cuphead.exe" and env.gamepad_emulator is not None:
    print(f"GamepadEnv ready for {args.process} at {env.env_fps} FPS")
    input("Press enter to create a virtual controller and start rollouts...")
    for i in range(3):
        print(f"{3 - i}...")
        time.sleep(1)

    def press(button):
        env.gamepad_emulator.press_button(button)
        env.gamepad_emulator.gamepad.update()
        time.sleep(0.05)
        env.gamepad_emulator.release_button(button)
        env.gamepad_emulator.gamepad.update()

    press("SOUTH")
    for k in range(5):
        press("EAST")
        time.sleep(0.3)

env.reset()
env.pause()

# Start System 2 Ollama Integration
system2 = OllamaSystem2(model_name="llama3.2:1b", interval_s=1.0)
system2.start()

def _pulse_controller_button(button_name: str, pulse_ms: int):
    button_name = str(button_name).strip().upper()
    pulse_s = max(0.02, float(pulse_ms) / 1000.0)
    try:
        env.gamepad_emulator.press_button(button_name)
        env.gamepad_emulator.gamepad.update()
        time.sleep(pulse_s)
        env.gamepad_emulator.release_button(button_name)
        env.gamepad_emulator.gamepad.update()
    except Exception as e:
        print(f"[warn] controller pulse failed ({e!r}); trying reconnect...")
        try:
            env.gamepad_emulator.reconnect()
            env.gamepad_emulator.press_button(button_name)
            env.gamepad_emulator.gamepad.update()
            time.sleep(pulse_s)
            env.gamepad_emulator.release_button(button_name)
            env.gamepad_emulator.gamepad.update()
        except Exception as e2:
            print(f"[warn] controller pulse retry failed ({e2!r})")

# Optional controller sanity test before letting the model run.
if args.test_controller:
    if env.gamepad_emulator is None:
        raise SystemExit("--test-controller requires --input-backend controller (ViGEm).")
    print("Controller test sequence (watch the game):")
    input("- Press Enter: left stick full right for ~1s")
    test_action = zero_action.copy()
    test_action["AXIS_LEFTX"] = np.array([32767], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: left stick full UP for ~1s")
    test_action = zero_action.copy()
    test_action["AXIS_LEFTY"] = np.array([-32768], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: left stick full DOWN for ~1s")
    test_action = zero_action.copy()
    test_action["AXIS_LEFTY"] = np.array([32767], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: right stick full right for ~1s (camera should pan)")
    test_action = zero_action.copy()
    test_action["AXIS_RIGHTX"] = np.array([32767], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: right stick full UP for ~1s (camera should look up)")
    test_action = zero_action.copy()
    test_action["AXIS_RIGHTY"] = np.array([-32768], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: right stick full DOWN for ~1s (camera should look down)")
    test_action = zero_action.copy()
    test_action["AXIS_RIGHTY"] = np.array([32767], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: press SOUTH/A a few times")
    for _ in range(5):
        press_action = zero_action.copy()
        press_action["SOUTH"] = 1
        obs, reward, terminated, truncated, info = env.step(action=press_action, step_duration=0.06)
        obs, reward, terminated, truncated, info = env.step(action=zero_action, step_duration=0.06)

    input("- Press Enter: hold RIGHT_TRIGGER for ~1s")
    trig_action = zero_action.copy()
    trig_action["RIGHT_TRIGGER"] = np.array([255], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=trig_action, step_duration=0.02)

    input("Press Enter to continue into model loop...")

# Optionally force the game to switch to controller prompts/mode.
if args.force_controller_pulse_on_start:
    if env.gamepad_emulator is None:
        print("[warn] --force-controller-pulse-on-start ignored (no controller in --input-backend kbm mode)")
    else:
        _pulse_controller_button(args.force_controller_button, args.force_controller_pulse_ms)

# Initial call to get state
if args.input_backend == "kbm":
    obs = env.render()
    reward = 0.0
    terminated = False
    truncated = False
    info = {}
else:
    obs, reward, terminated, truncated, info = env.step(action=zero_action)

frames = None
step_count = 0
controller_steps = 0

with VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="medium") as debug_recorder:
    with VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="medium") as clean_recorder:
        try:
            while True:
                if _override_pressed():
                    obs = env.render()
                    time.sleep(1.0 / max(1, args.env_fps))
                    continue

                obs = preprocess_img(obs)
                obs.save(PATH_DEBUG / f"{step_count:05d}.png")

                try:
                    # System 2 injection: Get the latest strategy from Ollama thread
                    # and push it to the policy inference if the model supports it.
                    # Currently NitroGen models use 'game' strings to lookup game_ids.
                    # We can pass the strategy string into predict.
                    current_strategy = system2.get_strategy()

                    t_req0 = time.perf_counter()
                    pred = policy.predict(obs, game=current_strategy)
                    t_req1 = time.perf_counter()
                    t_resp = time.time()
                except Exception as e:
                    print(f"[warn] predict failed ({e!r}); reconnecting and retrying...")
                    policy.reconnect()
                    time.sleep(0.5)
                    try:
                        policy.reset()
                    except Exception:
                        pass
                    if args.input_backend == "kbm":
                        obs = env.render()
                        reward = 0.0
                        terminated = False
                        truncated = False
                        info = {}
                    else:
                        obs, reward, terminated, truncated, info = env.step(action=zero_action)
                    continue

                j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]
                buttons_raw = pred.get("buttons_raw", None)
                j_left_raw = pred.get("j_left_raw", None)
                j_right_raw = pred.get("j_right_raw", None)
                meta = pred.get("_meta") if isinstance(pred, dict) else None
                if isinstance(meta, dict):
                    cmd_id = meta.get("command_id")
                    if isinstance(cmd_id, int):
                        last_cmd_id = cmd_id
                        if expected_command_id is None:
                            expected_command_id = cmd_id
                        if cmd_id != expected_command_id:
                            print(f"[warn] command_id jump: got={cmd_id} expected={expected_command_id}")
                            expected_command_id = cmd_id
                        expected_command_id += 1
                    if args.debug:
                        ts = time.strftime("%H:%M:%S")
                        rtt_s = t_req1 - t_req0
                        infer_s = meta.get("server_infer_time_s")
                        recv_s = meta.get("server_recv_time_s")
                        send_s = meta.get("server_send_time_s")
                        resp_s = float(t_resp) if isinstance(t_resp, (int, float)) else None
                        print(
                            f"[{ts}] [cmd {meta.get('command_id')}] rtt={rtt_s:.3f}s "
                            f"server_infer={infer_s:.3f}s recv={recv_s:.6f} send={send_s:.6f} resp={resp_s:.6f}"
                            if isinstance(infer_s, (int, float)) and isinstance(recv_s, (int, float)) and isinstance(send_s, (int, float)) and isinstance(resp_s, (int, float))
                            else f"[{ts}] [cmd {meta.get('command_id')}] rtt={rtt_s:.3f}s"
                        )

                n = len(buttons)
                assert n == len(j_left) == len(j_right), "Mismatch in action lengths"

                if args.trace_model:
                    for i in range(n):
                        jl_i = j_left_raw[i] if isinstance(j_left_raw, np.ndarray) else j_left[i]
                        jr_i = j_right_raw[i] if isinstance(j_right_raw, np.ndarray) else j_right[i]
                        br_i = buttons_raw[i] if isinstance(buttons_raw, np.ndarray) else None
                        print(_format_model_trace(i, jl_i, jr_i, br_i, TOKEN_SET, args.trace_model_topk))

                env_actions: list[OrderedDict] = []
                kbm_actions: list[dict] = []

                right_deadzone = float(args.right_stick_deadzone)
                right_gain = float(args.right_stick_gain) if args.right_stick_gain is not None else float(args.stick_gain)

                for i in range(n):
                    move_action = zero_action.copy()

                    xl, yl = float(j_left[i][0]), float(j_left[i][1])
                    xr, yr = float(j_right[i][0]), float(j_right[i][1])
                    if args.invert_left_y:
                        yl = -yl
                    if args.invert_right_y:
                        yr = -yr

                    def _dz_gain(v: float, *, deadzone: float, gain: float) -> float:
                        v *= float(gain)
                        if abs(v) < float(deadzone):
                            return 0.0
                        return float(np.clip(v, -1.0, 1.0))

                    xl = _dz_gain(xl, deadzone=float(args.stick_deadzone), gain=float(args.stick_gain))
                    yl = _dz_gain(yl, deadzone=float(args.stick_deadzone), gain=float(args.stick_gain))
                    xr = _dz_gain(xr, deadzone=right_deadzone, gain=right_gain)
                    yr = _dz_gain(yr, deadzone=right_deadzone, gain=right_gain)

                    # Use raw scores when available so --button-threshold actually works.
                    button_vector_raw = buttons_raw[i] if isinstance(buttons_raw, np.ndarray) else buttons[i]
                    assert len(button_vector_raw) == len(TOKEN_SET), "Button vector length does not match token set length"

                    if IS_KEYBOARD_MODEL:
                        if kbm is None:
                            raise RuntimeError("IS_KEYBOARD_MODEL=True but KBM emulator is not initialized.")

                        pressed_names = [name for name, value in zip(TOKEN_SET, button_vector_raw) if float(value) > BUTTON_PRESS_THRES]
                        desired_keys: set[str] = set()
                        desired_lmb = False
                        desired_rmb = False
                        desired_mmb = False
                        desired_x1 = False
                        desired_x2 = False
                        wheel = 0
                        for name in pressed_names:
                            nrm = str(name).strip().lower()
                            if nrm in ("left_click", "lmb"):
                                desired_lmb = True
                            elif nrm in ("right_click", "rmb"):
                                desired_rmb = True
                            elif nrm in ("middle_click", "mmb"):
                                desired_mmb = True
                            elif nrm in ("mouse_x1", "x1", "mb4"):
                                desired_x1 = True
                            elif nrm in ("mouse_x2", "x2", "mb5"):
                                desired_x2 = True
                            elif nrm == "scroll_up":
                                wheel += 1
                            elif nrm == "scroll_down":
                                wheel -= 1
                            else:
                                desired_keys.add(nrm)

                        # Movement keys from model sticks (so movement works even if the model doesn't press WASD buttons).
                        move_dz = float(args.kbm_move_deadzone)
                        up_key = "w" if "w" in TOKEN_SET_LOWER else ("arrow_up" if "arrow_up" in TOKEN_SET_LOWER else "w")
                        down_key = "s" if "s" in TOKEN_SET_LOWER else ("arrow_down" if "arrow_down" in TOKEN_SET_LOWER else "s")
                        left_key = "a" if "a" in TOKEN_SET_LOWER else ("arrow_left" if "arrow_left" in TOKEN_SET_LOWER else "a")
                        right_key = "d" if "d" in TOKEN_SET_LOWER else ("arrow_right" if "arrow_right" in TOKEN_SET_LOWER else "d")
                        # Y: negative=forward after inversion step above.
                        if yl <= -move_dz:
                            desired_keys.add(up_key)
                            desired_keys.discard(down_key)
                        elif yl >= move_dz:
                            desired_keys.add(down_key)
                            desired_keys.discard(up_key)
                        else:
                            desired_keys.discard(up_key)
                            desired_keys.discard(down_key)
                        if xl <= -move_dz:
                            desired_keys.add(left_key)
                            desired_keys.discard(right_key)
                        elif xl >= move_dz:
                            desired_keys.add(right_key)
                            desired_keys.discard(left_key)
                        else:
                            desired_keys.discard(left_key)
                            desired_keys.discard(right_key)

                        if NO_MENU:
                            desired_keys.discard("esc")
                            desired_keys.discard("escape")
                            desired_keys.discard("enter")

                        dx = int(xr * float(args.kbm_mouse_speed))
                        dy = int(yr * float(args.kbm_mouse_speed))
                        if abs(xr) < float(args.kbm_mouse_deadzone):
                            dx = 0
                        if abs(yr) < float(args.kbm_mouse_deadzone):
                            dy = 0
                        kbm_actions.append(
                            {
                                "keys": sorted(desired_keys),
                                "lmb": bool(desired_lmb),
                                "rmb": bool(desired_rmb),
                                "mmb": bool(desired_mmb),
                                "x1": bool(desired_x1),
                                "x2": bool(desired_x2),
                                "wheel": int(wheel),
                                "mouse_dx": int(dx),
                                "mouse_dy": int(dy),
                            }
                        )
                    else:
                        move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.int32)
                        move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.int32)
                        move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.int32)
                        move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.int32)

                        for name, value in zip(TOKEN_SET, button_vector_raw):
                            if "TRIGGER" in name:
                                v = float(np.clip(float(value), 0.0, 1.0))
                                move_action[name] = np.array([int(v * 255)], dtype=np.int32)
                            else:
                                move_action[name] = 1 if float(value) > BUTTON_PRESS_THRES else 0

                        if args.disable_dpad:
                            move_action["DPAD_UP"] = 0
                            move_action["DPAD_RIGHT"] = 0
                            move_action["DPAD_DOWN"] = 0
                            move_action["DPAD_LEFT"] = 0

                        env_actions.append(move_action)

                cmd_txt = f" cmd={last_cmd_id}" if isinstance(last_cmd_id, int) else ""
                to_run = kbm_actions if IS_KEYBOARD_MODEL else env_actions
                print(f"Executing {len(to_run)} actions, each action will be repeated {action_downsample_ratio} times{cmd_txt}")

                action_hold_s = max(0.0, args.action_hold_ms / 1000.0)
                exec_start_s = time.time()
                if args.debug and isinstance(last_cmd_id, int) and isinstance(meta, dict):
                    ts = time.strftime("%H:%M:%S")
                    send_s = meta.get("server_send_time_s")
                    if isinstance(send_s, (int, float)):
                        print(f"[{ts}] [cmd {last_cmd_id}] exec_start={exec_start_s:.6f} since_send={exec_start_s - float(send_s):.3f}s")

                for i in range(len(to_run)):
                    a = to_run[i]

                    if NO_MENU and (not IS_KEYBOARD_MODEL):
                        try:
                            if a.get("START"):
                                print("Model predicted start, disabling this action")
                            a["GUIDE"] = 0
                            a["START"] = 0
                            a["BACK"] = 0
                        except Exception:
                            pass

                    if args.debug and (not IS_KEYBOARD_MODEL):
                        print(_format_sub_action(i, a, TOKEN_SET, cmd_id=last_cmd_id))
                    if args.debug and IS_KEYBOARD_MODEL:
                        print(
                            f"[cmd {last_cmd_id}] [{i:02d}] keys={a.get('keys')} "
                            f"lmb={a.get('lmb')} rmb={a.get('rmb')} mmb={a.get('mmb')} x1={a.get('x1')} x2={a.get('x2')} "
                            f"wheel={a.get('wheel')} mouse=({a.get('mouse_dx')},{a.get('mouse_dy')})"
                        )

                    for _ in range(action_downsample_ratio):
                        dur = max(env.step_duration, action_hold_s) if action_hold_s > 0 else env.step_duration
                        if args.input_backend == "kbm" and kbm is not None and IS_KEYBOARD_MODEL:
                            kbm.step_keys(
                                keys_down=set(a.get("keys") or []),
                                lmb=bool(a.get("lmb")),
                                rmb=bool(a.get("rmb")),
                                mmb=bool(a.get("mmb")),
                                x1=bool(a.get("x1")),
                                x2=bool(a.get("x2")),
                                wheel=int(a.get("wheel") or 0),
                                mouse_dx=int(a.get("mouse_dx") or 0),
                                mouse_dy=int(a.get("mouse_dy") or 0),
                                duration_s=dur,
                            )
                            obs = env.render()
                        elif args.input_backend == "kbm" and kbm is not None:
                            kbm.step(a, duration_s=dur)
                            obs = env.render()
                        else:
                            if (
                                env.gamepad_emulator is not None
                                and args.controller_keepalive_every > 0
                                and controller_steps > 0
                                and (controller_steps % args.controller_keepalive_every == 0)
                            ):
                                env.gamepad_emulator.reset()
                            obs, reward, terminated, truncated, info = env.step(action=a, step_duration=dur)

                        controller_steps += 1

                        # resize obs to 720p
                        obs_viz = np.array(obs).copy()
                        clean_viz = cv2.resize(obs_viz, (1920, 1080), interpolation=cv2.INTER_AREA)
                        debug_viz = create_viz(
                            cv2.resize(obs_viz, (1280, 720), interpolation=cv2.INTER_AREA), # 720p
                            i,
                            j_left,
                            j_right,
                            buttons,
                            token_set=TOKEN_SET
                        )
                        debug_recorder.add_frame(debug_viz)
                        clean_recorder.add_frame(clean_viz)

                exec_end_s = time.time()
                if args.debug and isinstance(last_cmd_id, int):
                    ts = time.strftime("%H:%M:%S")
                    print(f"[{ts}] [cmd {last_cmd_id}] exec_end={exec_end_s:.6f} exec_dur={exec_end_s - exec_start_s:.3f}s")

                # Append executed actions to JSONL file
                with open(PATH_ACTIONS, "a", encoding="utf-8") as f:
                    if IS_KEYBOARD_MODEL:
                        for i, a in enumerate(kbm_actions):
                            obj = {
                                "step": int(step_count),
                                "substep": int(i),
                                "cmd_id": int(last_cmd_id) if isinstance(last_cmd_id, int) else None,
                                "keys": list(a.get("keys") or []),
                                "lmb": bool(a.get("lmb")),
                                "rmb": bool(a.get("rmb")),
                                "mmb": bool(a.get("mmb")),
                                "x1": bool(a.get("x1")),
                                "x2": bool(a.get("x2")),
                                "wheel": int(a.get("wheel") or 0),
                                "mouse_dx": int(a.get("mouse_dx") or 0),
                                "mouse_dy": int(a.get("mouse_dy") or 0),
                            }
                            json.dump(obj, f)
                            f.write("\n")
                    else:
                        for i, a in enumerate(env_actions):
                            # convert numpy arrays to lists for JSON serialization
                            for k, v in a.items():
                                if isinstance(v, np.ndarray):
                                    a[k] = v.tolist()
                            a["step"] = step_count
                            a["substep"] = i
                            json.dump(a, f)
                            f.write("\n")

                if args.force_controller_pulse_every and args.force_controller_pulse_every > 0:
                    if (step_count % int(args.force_controller_pulse_every)) == 0:
                        if env.gamepad_emulator is None:
                            pass
                        else:
                            _pulse_controller_button(args.force_controller_button, args.force_controller_pulse_ms)

                step_count += 1
        finally:
            if kbm is not None:
                try:
                    kbm.release_all()
                except Exception:
                    pass
            env.unpause()
            env.close()
