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

try:
    import win32api
    import win32con
except Exception:
    win32api = None
    win32con = None

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
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
parser.add_argument("--stick-deadzone", type=float, default=0.08, help="Joystick deadzone in normalized units")
parser.add_argument("--stick-gain", type=float, default=1.0, help="Multiply joystick magnitude before clipping")
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

args = parser.parse_args()

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

TOKEN_SET = BUTTON_ACTION_TOKENS


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


def _format_sub_action(idx: int, move_action: OrderedDict, token_set) -> str:
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
    return f"[{idx:02d}] LX={lx:6d} LY={ly:6d} RX={rx:6d} RY={ry:6d} LT={lt:3d} RT={rt:3d} btn={buttons_s}"


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
    screenshot_backend=args.screenshot_backend,
    capture_margin=args.capture_margin,
    reuse_last_frame_on_capture_fail=args.reuse_last_frame_on_fail,
    capture_retry_delay_s=args.capture_retry_delay_ms / 1000.0,
    hard_fail_after=args.capture_hard_fail_after,
    hard_fail_sleep_s=args.capture_hard_fail_sleep_ms / 1000.0,
)

# These games requires to open a menu to initialize the controller
if args.process == "isaac-ng.exe":
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

if args.process == "Cuphead.exe":
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
    print("Controller test sequence (watch the game):")
    input("- Press Enter: left stick full right for ~1s")
    test_action = zero_action.copy()
    test_action["AXIS_LEFTX"] = np.array([32767], dtype=np.int32)
    for _ in range(max(1, int(args.env_fps))):
        obs, reward, terminated, truncated, info = env.step(action=test_action, step_duration=0.02)

    input("- Press Enter: right stick full right for ~1s (camera should pan)")
    test_action = zero_action.copy()
    test_action["AXIS_RIGHTX"] = np.array([32767], dtype=np.int32)
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
    _pulse_controller_button(args.force_controller_button, args.force_controller_pulse_ms)

# Initial call to get state
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
                    pred = policy.predict(obs)
                except Exception as e:
                    print(f"[warn] predict failed ({e!r}); reconnecting and retrying...")
                    policy.reconnect()
                    time.sleep(0.5)
                    try:
                        policy.reset()
                    except Exception:
                        pass
                    obs, reward, terminated, truncated, info = env.step(action=zero_action)
                    continue

                j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]
                buttons_raw = pred.get("buttons_raw", None)
                j_left_raw = pred.get("j_left_raw", None)
                j_right_raw = pred.get("j_right_raw", None)

                n = len(buttons)
                assert n == len(j_left) == len(j_right), "Mismatch in action lengths"

                if args.trace_model:
                    for i in range(n):
                        jl_i = j_left_raw[i] if isinstance(j_left_raw, np.ndarray) else j_left[i]
                        jr_i = j_right_raw[i] if isinstance(j_right_raw, np.ndarray) else j_right[i]
                        br_i = buttons_raw[i] if isinstance(buttons_raw, np.ndarray) else None
                        print(_format_model_trace(i, jl_i, jr_i, br_i, TOKEN_SET, args.trace_model_topk))

                env_actions = []

                for i in range(n):
                    move_action = zero_action.copy()

                    xl, yl = float(j_left[i][0]), float(j_left[i][1])
                    xr, yr = float(j_right[i][0]), float(j_right[i][1])

                    def _dz_gain(v: float) -> float:
                        v *= float(args.stick_gain)
                        if abs(v) < float(args.stick_deadzone):
                            return 0.0
                        return float(np.clip(v, -1.0, 1.0))

                    xl = _dz_gain(xl)
                    yl = _dz_gain(yl)
                    xr = _dz_gain(xr)
                    yr = _dz_gain(yr)

                    move_action["AXIS_LEFTX"] = np.array([int(xl * 32767)], dtype=np.int32)
                    move_action["AXIS_LEFTY"] = np.array([int(yl * 32767)], dtype=np.int32)
                    move_action["AXIS_RIGHTX"] = np.array([int(xr * 32767)], dtype=np.int32)
                    move_action["AXIS_RIGHTY"] = np.array([int(yr * 32767)], dtype=np.int32)
                    
                    button_vector = buttons[i]
                    assert len(button_vector) == len(TOKEN_SET), "Button vector length does not match token set length"

                    
                    for name, value in zip(TOKEN_SET, button_vector):
                        if "TRIGGER" in name:
                            move_action[name] = np.array([int(value * 255)], dtype=np.int32)
                        else:
                            move_action[name] = 1 if value > BUTTON_PRESS_THRES else 0


                    env_actions.append(move_action)

                print(f"Executing {len(env_actions)} actions, each action will be repeated {action_downsample_ratio} times")

                action_hold_s = max(0.0, args.action_hold_ms / 1000.0)

                for i, a in enumerate(env_actions):
                    if NO_MENU:
                        if a["START"]:
                            print("Model predicted start, disabling this action")
                        a["GUIDE"] = 0
                        a["START"] = 0
                        a["BACK"] = 0

                    if args.debug:
                        print(_format_sub_action(i, a, TOKEN_SET))

                    for _ in range(action_downsample_ratio):
                        if args.controller_keepalive_every > 0 and controller_steps > 0 and (controller_steps % args.controller_keepalive_every == 0):
                            env.gamepad_emulator.reset()
                        obs, reward, terminated, truncated, info = env.step(
                            action=a,
                            step_duration=max(env.step_duration, action_hold_s) if action_hold_s > 0 else None,
                        )
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

                # Append env_actions dictionnary to JSONL file
                with open(PATH_ACTIONS, "a") as f:
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
                        _pulse_controller_button(args.force_controller_button, args.force_controller_pulse_ms)

                step_count += 1
        finally:
            env.unpause()
            env.close()
