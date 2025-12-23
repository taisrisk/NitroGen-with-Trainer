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

import argparse
parser = argparse.ArgumentParser(description="VLM Inference")
parser.add_argument("--process", type=str, default="celeste.exe", help="Game to play")
parser.add_argument("--allow-menu", action="store_true", help="Allow menu actions (Disabled by default)")
parser.add_argument("--port", type=int, default=5555, help="Port for model server")
parser.add_argument("--timeout-ms", type=int, default=120_000, help="Model server receive timeout (ms)")
parser.add_argument("--focus-each-step", action="store_true", help="Refocus the game window each step (helps some games accept virtual controller)")
parser.add_argument("--env-fps", type=int, default=30, help="Gamepad action FPS (lower improves capture stability)")
parser.add_argument("--obs-width", type=int, default=1280, help="Captured observation width (resized)")
parser.add_argument("--obs-height", type=int, default=720, help="Captured observation height (resized)")
parser.add_argument("--screenshot-backend", choices=["dxcam", "pyautogui"], default="dxcam", help="Screenshot backend")
parser.add_argument("--capture-margin", type=int, default=0, help="Crop margin (pixels) from each window edge before capture")
parser.add_argument("--reuse-last-frame-on-fail", action=argparse.BooleanOptionalAction, default=False, help="If capture fails, reuse the previous frame instead of waiting")
parser.add_argument("--capture-retry-delay-ms", type=int, default=20, help="Delay between capture retries (ms)")
parser.add_argument("--capture-hard-fail-after", type=int, default=5, help="After N consecutive capture fails, pause actions more aggressively")
parser.add_argument("--capture-hard-fail-sleep-ms", type=int, default=1000, help="Extra sleep after hard-fail threshold (ms)")
parser.add_argument("--test-controller", action="store_true", help="Interactive controller sanity test before running the model loop")

args = parser.parse_args()

policy = ModelClient(port=args.port, timeout_ms=args.timeout_ms)
policy.reset()
policy_info = policy.info()
action_downsample_ratio = policy_info["action_downsample_ratio"]

CKPT_NAME = Path(policy_info["ckpt_path"]).stem
NO_MENU = not args.allow_menu

PATH_DEBUG = PATH_REPO / "debug"
PATH_DEBUG.mkdir(parents=True, exist_ok=True)

PATH_OUT = (PATH_REPO / "out" / CKPT_NAME).resolve()
PATH_OUT.mkdir(parents=True, exist_ok=True)

BUTTON_PRESS_THRES = 0.5

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

# Optional controller sanity test before letting the model run.
if args.test_controller:
    input("Press Enter to test left stick (should move camera/menu)...")
    test_action = zero_action.copy()
    test_action["AXIS_LEFTX"] = np.array([32767], dtype=np.int32)
    for _ in range(60):
        obs, reward, terminated, truncated, info = env.step(action=test_action)
    input("Press Enter to continue into model loop...")

# Initial call to get state
obs, reward, terminated, truncated, info = env.step(action=zero_action)

frames = None
step_count = 0

with VideoRecorder(str(PATH_MP4_DEBUG), fps=60, crf=32, preset="medium") as debug_recorder:
    with VideoRecorder(str(PATH_MP4_CLEAN), fps=60, crf=28, preset="medium") as clean_recorder:
        try:
            while True:
                obs = preprocess_img(obs)
                obs.save(PATH_DEBUG / f"{step_count:05d}.png")

                pred = policy.predict(obs)

                j_left, j_right, buttons = pred["j_left"], pred["j_right"], pred["buttons"]

                n = len(buttons)
                assert n == len(j_left) == len(j_right), "Mismatch in action lengths"


                env_actions = []

                for i in range(n):
                    move_action = zero_action.copy()

                    xl, yl = j_left[i]
                    xr, yr = j_right[i]
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

                for i, a in enumerate(env_actions):
                    if NO_MENU:
                        if a["START"]:
                            print("Model predicted start, disabling this action")
                        a["GUIDE"] = 0
                        a["START"] = 0
                        a["BACK"] = 0

                    for _ in range(action_downsample_ratio):
                        obs, reward, terminated, truncated, info = env.step(action=a)

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


                step_count += 1
        finally:
            env.unpause()
            env.close()
