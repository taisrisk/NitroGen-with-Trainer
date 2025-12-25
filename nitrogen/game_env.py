import time
import platform

import pyautogui
import dxcam
import pywinctl as pwc
import xspeedhack as xsh
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from PIL import Image

import time

import vgamepad as vg

import psutil

assert platform.system().lower() == "windows", "This module is only supported on Windows."
import win32process
import win32gui
import win32api
import win32con

def get_process_info(process_name):
    """
    Get process information for a given process name on Windows.
    
    Args:
        process_name (str): Name of the process (e.g., "isaac-ng.exe")
    
    Returns:
        list: List of dictionaries containing PID, window_name, and architecture
              for each matching process. Returns empty list if no process found.
    """
    results = []
    
    # Find all processes with the given name
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'].lower() == process_name.lower():
                pid = proc.info['pid']
                
                # Get architecture
                try:
                    # Check if process is 32-bit or 64-bit
                    process_handle = win32api.OpenProcess(
                        win32con.PROCESS_QUERY_INFORMATION, 
                        False, 
                        pid
                    )
                    is_wow64 = win32process.IsWow64Process(process_handle)
                    win32api.CloseHandle(process_handle)
                    
                    # On 64-bit Windows: WOW64 means "Windows 32-bit on Windows 64-bit", i.e. a 32-bit process
                    architecture = "x86" if is_wow64 else "x64"
                except:
                    architecture = "unknown"
                
                # Find windows associated with this PID
                windows = []
                
                def enum_window_callback(hwnd, pid_to_find):
                    _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
                    if found_pid == pid_to_find:
                        window_text = win32gui.GetWindowText(hwnd)
                        if window_text and win32gui.IsWindowVisible(hwnd):
                            windows.append({
                                'hwnd': hwnd,
                                'title': window_text,
                                'visible': win32gui.IsWindowVisible(hwnd)
                            })
                    return True
                
                # Find all windows for this PID
                try:
                    win32gui.EnumWindows(enum_window_callback, pid)
                except:
                    pass
                
                # Choose the best window
                window_name = None
                if windows:
                    if len(windows) > 1:
                        print(f"Multiple windows found for PID {pid}: {[win['title'] for win in windows]}")
                        print("Using heuristics to select the correct window...")
                    # Filter out common proxy/helper windows
                    proxy_keywords = ['d3dproxywindow', 'proxy', 'helper', 'overlay']
                    
                    # First try to find a visible window without proxy keywords
                    for win in windows:
                        if not any(keyword in win['title'].lower() for keyword in proxy_keywords):
                            window_name = win['title']
                            break
                    
                    # If no good window found, just use the first one
                    if window_name is None and windows:
                        window_name = windows[0]['title']
                
                results.append({
                    'pid': pid,
                    'window_name': window_name,
                    'architecture': architecture
                })
                
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if len(results) == 0:
        raise ValueError(f"No process found with name: {process_name}")
    elif len(results) > 1:
        print(f"Warning: Multiple processes found with name '{process_name}'. Returning first match.")
    
    return results[0]


XBOX_MAPPING = {
    "DPAD_UP": "XUSB_GAMEPAD_DPAD_UP",
    "DPAD_DOWN": "XUSB_GAMEPAD_DPAD_DOWN",
    "DPAD_LEFT": "XUSB_GAMEPAD_DPAD_LEFT",
    "DPAD_RIGHT": "XUSB_GAMEPAD_DPAD_RIGHT",
    "START": "XUSB_GAMEPAD_START",
    "BACK": "XUSB_GAMEPAD_BACK",
    "LEFT_SHOULDER": "XUSB_GAMEPAD_LEFT_SHOULDER",
    "RIGHT_SHOULDER": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "GUIDE": "XUSB_GAMEPAD_GUIDE",
    "WEST": "XUSB_GAMEPAD_X",
    "SOUTH": "XUSB_GAMEPAD_A",
    "EAST": "XUSB_GAMEPAD_B",
    "NORTH": "XUSB_GAMEPAD_Y",
    "LEFT_TRIGGER": "LEFT_TRIGGER",
    "RIGHT_TRIGGER": "RIGHT_TRIGGER",
    "AXIS_LEFTX": "LEFT_JOYSTICK",
    "AXIS_LEFTY": "LEFT_JOYSTICK",
    "AXIS_RIGHTX": "RIGHT_JOYSTICK",
    "AXIS_RIGHTY": "RIGHT_JOYSTICK",
    "LEFT_THUMB": "XUSB_GAMEPAD_LEFT_THUMB",
    "RIGHT_THUMB": "XUSB_GAMEPAD_RIGHT_THUMB",
}

PS4_MAPPING = {
    "DPAD_UP": "DS4_BUTTON_DPAD_NORTH",
    "DPAD_DOWN": "DS4_BUTTON_DPAD_SOUTH",
    "DPAD_LEFT": "DS4_BUTTON_DPAD_WEST",
    "DPAD_RIGHT": "DS4_BUTTON_DPAD_EAST",
    "START": "DS4_BUTTON_OPTIONS",
    "BACK": "DS4_BUTTON_SHARE",
    "LEFT_SHOULDER": "DS4_BUTTON_SHOULDER_LEFT",
    "RIGHT_SHOULDER": "DS4_BUTTON_SHOULDER_RIGHT",
    "GUIDE": "DS4_BUTTON_GUIDE",
    "WEST": "DS4_BUTTON_SQUARE",
    "SOUTH": "DS4_BUTTON_CROSS",
    "EAST": "DS4_BUTTON_CIRCLE",
    "NORTH": "DS4_BUTTON_TRIANGLE",
    "LEFT_TRIGGER": "LEFT_TRIGGER",
    "RIGHT_TRIGGER": "RIGHT_TRIGGER",
    "AXIS_LEFTX": "LEFT_JOYSTICK",
    "AXIS_LEFTY": "LEFT_JOYSTICK",
    "AXIS_RIGHTX": "RIGHT_JOYSTICK",
    "AXIS_RIGHTY": "RIGHT_JOYSTICK",
    "LEFT_THUMB": "DS4_BUTTON_THUMB_LEFT",
    "RIGHT_THUMB": "DS4_BUTTON_THUMB_RIGHT",
}


class GamepadEmulator:
    def __init__(self, controller_type="xbox", system="windows", *, invert_y_axis: bool = False):
        """
        Initialize the GamepadEmulator with a specific controller type and system.

        Parameters:
        controller_type (str): The type of controller to emulate ("xbox" or "ps4").
        system (str): The operating system to use, which affects joystick value handling.
        """
        self.controller_type = controller_type
        self.system = system
        self.invert_y_axis = bool(invert_y_axis)
        self._create_gamepad()

        # Initialize joystick values to keep track of the current state
        self.left_joystick_x: int = 0
        self.left_joystick_y: int = 0
        self.right_joystick_x: int = 0
        self.right_joystick_y: int = 0
        self._consecutive_failures: int = 0
        self._last_failure: str | None = None

    def _create_gamepad(self):
        if self.controller_type == "xbox":
            try:
                self.gamepad = vg.VX360Gamepad()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create VX360Gamepad (ViGEm). Is ViGEmBus installed? ({e!r})"
                ) from e
            self.mapping = XBOX_MAPPING
        elif self.controller_type == "ps4":
            try:
                self.gamepad = vg.VDS4Gamepad()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to create VDS4Gamepad (ViGEm). Is ViGEmBus installed? ({e!r})"
                ) from e
            self.mapping = PS4_MAPPING
        else:
            raise ValueError("Unsupported controller type")

    def reconnect(self):
        self._create_gamepad()
        self.reset()

    def _record_failure(self, where: str, exc: Exception | None = None) -> None:
        self._consecutive_failures += 1
        self._last_failure = where
        if exc is not None:
            print(f"[warn] controller {where} failed (n={self._consecutive_failures}): {exc!r}")
        else:
            print(f"[warn] controller {where} failed (n={self._consecutive_failures})")

    def _clear_failures(self) -> None:
        self._consecutive_failures = 0
        self._last_failure = None

    def _maybe_reconnect(self, where: str, exc: Exception) -> None:
        self._record_failure(where, exc)
        # Avoid destroying/recreating the virtual controller on the first transient error.
        if self._consecutive_failures < 3:
            return
        print(f"[warn] controller reconnect (last_failure={self._last_failure!r})")
        try:
            self._create_gamepad()
        except Exception as e:
            # If we can't recreate the device, keep the existing one and surface the root error.
            raise RuntimeError(f"Failed to recreate virtual controller after {where} failure ({e!r})") from exc
        self._clear_failures()

    def step(self, action):
        """
        Perform actions based on the provided action dictionary.

        Parameters:
        action (dict): Dictionary of an action to be performed. Keys are control names,
                       and values are their respective states.
        """
        try:
            self.gamepad.reset()
        except Exception as e:
            self._maybe_reconnect("reset", e)
            # Retry once; if it still fails, then force a reconnect.
            try:
                self.gamepad.reset()
            except Exception as e2:
                self._record_failure("reset_retry", e2)
                self.reconnect()
                self.gamepad.reset()

        # Handle buttons
        for control in [
            "EAST",
            "SOUTH",
            "NORTH",
            "WEST",
            "BACK",
            "GUIDE",
            "START",
            "DPAD_DOWN",
            "DPAD_LEFT",
            "DPAD_RIGHT",
            "DPAD_UP",
            "LEFT_SHOULDER",
            "RIGHT_SHOULDER",
            "LEFT_THUMB",
            "RIGHT_THUMB",
        ]:
            if control in action:
                if action[control]:
                    self.press_button(control)
                else:
                    self.release_button(control)

        # Handle triggers
        if "LEFT_TRIGGER" in action:
            self.set_trigger("LEFT_TRIGGER", action["LEFT_TRIGGER"][0])
        if "RIGHT_TRIGGER" in action:
            self.set_trigger("RIGHT_TRIGGER", action["RIGHT_TRIGGER"][0])

        # Handle joysticks
        if "AXIS_LEFTX" in action and "AXIS_LEFTY" in action:
            self.set_joystick("AXIS_LEFTX", action["AXIS_LEFTX"][0])
            self.set_joystick("AXIS_LEFTY", action["AXIS_LEFTY"][0])

        if "AXIS_RIGHTX" in action and "AXIS_RIGHTY" in action:
            self.set_joystick("AXIS_RIGHTX", action["AXIS_RIGHTX"][0])
            self.set_joystick("AXIS_RIGHTY", action["AXIS_RIGHTY"][0])

        try:
            self.gamepad.update()
            self._clear_failures()
        except Exception as e:
            self._maybe_reconnect("update", e)
            # Retry once; if it still fails, then force a reconnect.
            try:
                self.gamepad.update()
                self._clear_failures()
            except Exception as e2:
                self._record_failure("update_retry", e2)
                self.reconnect()
                self.gamepad.update()
                self._clear_failures()

    def press_button(self, button):
        """
        Press a button on the gamepad.

        Parameters:
        button (str): The unified name of the button to press.
        """
        button_mapped = self.mapping.get(button)
        if self.controller_type == "xbox":
            self.gamepad.press_button(button=getattr(vg.XUSB_BUTTON, button_mapped))
        elif self.controller_type == "ps4":
            self.gamepad.press_button(button=getattr(vg.DS4_BUTTONS, button_mapped))
        else:
            raise ValueError("Unsupported controller type")

    def release_button(self, button):
        """
        Release a button on the gamepad.

        Parameters:
        button (str): The unified name of the button to release.
        """
        button_mapped = self.mapping.get(button)
        if self.controller_type == "xbox":
            self.gamepad.release_button(button=getattr(vg.XUSB_BUTTON, button_mapped))
        elif self.controller_type == "ps4":
            self.gamepad.release_button(button=getattr(vg.DS4_BUTTONS, button_mapped))
        else:
            raise ValueError("Unsupported controller type")

    def set_trigger(self, trigger, value):
        """
        Set the value of a trigger on the gamepad.

        Parameters:
        trigger (str): The unified name of the trigger.
        value (float): The value to set the trigger to (between 0 and 1).
        """
        value = int(value)
        value = max(0, min(255, value))
        trigger_mapped = self.mapping.get(trigger)
        if trigger_mapped == "LEFT_TRIGGER":
            self.gamepad.left_trigger(value=value)
        elif trigger_mapped == "RIGHT_TRIGGER":
            self.gamepad.right_trigger(value=value)
        else:
            raise ValueError("Unsupported trigger action")

    def set_joystick(self, joystick, value):
        """
        Set the position of a joystick on the gamepad.

        Parameters:
        joystick (str): The name of the joystick axis.
        value (float): The value to set the joystick axis to (between -32768 and 32767)
        """
        value = int(value)
        value = max(-32768, min(32767, value))
        if joystick == "AXIS_LEFTX":
            self.left_joystick_x = value
            self.gamepad.left_joystick(x_value=self.left_joystick_x, y_value=self.left_joystick_y)
        elif joystick == "AXIS_LEFTY":
            if self.invert_y_axis:
                value = -value - 1
            self.left_joystick_y = value
            self.gamepad.left_joystick(x_value=self.left_joystick_x, y_value=self.left_joystick_y)
        elif joystick == "AXIS_RIGHTX":
            self.right_joystick_x = value
            self.gamepad.right_joystick(
                x_value=self.right_joystick_x, y_value=self.right_joystick_y
            )
        elif joystick == "AXIS_RIGHTY":
            if self.invert_y_axis:
                value = -value - 1
            self.right_joystick_y = value
            self.gamepad.right_joystick(
                x_value=self.right_joystick_x, y_value=self.right_joystick_y
            )
        else:
            raise ValueError("Unsupported joystick action")

    def wakeup(self, duration=0.1):
        """
        Wake up the controller by pressing a button.

        Parameters:
        duration (float): Duration to press the button.
        """
        try:
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_A)
            self.gamepad.update()
            time.sleep(duration)
            self.gamepad.reset()
            self.gamepad.update()
            time.sleep(duration)
        except Exception:
            self.gamepad.press_button(vg.XUSB_BUTTON.XUSB_GAMEPAD_LEFT_THUMB)
            self.gamepad.update()
            time.sleep(duration)
            self.gamepad.reset()
            self.gamepad.update()
            time.sleep(duration)

    def reset(self):
        """
        Reset the gamepad to its default state.
        """
        try:
            self.gamepad.reset()
            self.gamepad.update()
            self._clear_failures()
        except Exception as e:
            self._maybe_reconnect("reset_call", e)
            # Retry once; if it still fails, then force a reconnect.
            try:
                self.gamepad.reset()
                self.gamepad.update()
                self._clear_failures()
            except Exception as e2:
                self._record_failure("reset_call_retry", e2)
                self.reconnect()
                self.gamepad.reset()
                self.gamepad.update()
                self._clear_failures()

class PyautoguiScreenshotBackend:

    def __init__(self, bbox):
        self.bbox = bbox

    def screenshot(self):
        return pyautogui.screenshot(region=self.bbox)

class DxcamScreenshotBackend:
    def __init__(self, bbox, reuse_last_on_fail: bool = True):
        import dxcam
        try:
            self.camera = dxcam.create(output_idx=0, output_color="RGB", max_buffer_len=1)
        except TypeError:
            try:
                self.camera = dxcam.create(output_idx=0, output_color="RGB")
            except TypeError:
                self.camera = dxcam.create()
        self.bbox = bbox
        self.last_screenshot = None
        self.reuse_last_on_fail = bool(reuse_last_on_fail)

    def screenshot(self):
        screenshot = self.camera.grab(region=self.bbox)
        if screenshot is None:
            if self.reuse_last_on_fail and self.last_screenshot is not None:
                return self.last_screenshot
            return None
        screenshot = Image.fromarray(screenshot)
        self.last_screenshot = screenshot
        return screenshot


class GamepadEnv(Env):
    """
    Base class for creating a game environment controlled with a gamepad.

    Attributes:
    game (str): Name of the game to interact with.
    image_height (int): Height of the observation space.
    image_width (int): Width of the observation space.
    controller_type (str): Platform for the gamepad emulator ("xbox" or "ps4").
    game_speed (float): Speed multiplier for the game.
    env_fps (int): Number of actions to perform per second at normal speed.
    async_mode (bool): Whether to pause/unpause the game during each step.
    """

    def __init__(
        self,
        game,
        image_height=1440,
        image_width=2560,
        controller_type="xbox",
        enable_controller: bool = True,
        game_speed=1.0,
        env_fps=10,
        async_mode=True,
        screenshot_backend="dxcam",
        focus_each_step: bool = False,
        capture_margin: int = 0,
        reuse_last_frame_on_capture_fail: bool = True,
        capture_retry_delay_s: float = 0.02,
        hard_fail_after: int = 5,
        hard_fail_sleep_s: float = 1.0,
        invert_y_axis: bool = False,
    ):
        super().__init__()

        # Assert that system is windows
        os_name = platform.system().lower()
        assert os_name == "windows", "This environment is currently only supported on Windows."
        assert controller_type in ["xbox", "ps4"], "Platform must be either 'xbox' or 'ps4'"
        assert screenshot_backend in ["pyautogui", "dxcam"], "Screenshot backend must be either 'pyautogui' or 'dxcam'"

        self.game = game
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.game_speed = game_speed
        self.env_fps = env_fps
        self.step_duration = self.calculate_step_duration()
        self.async_mode = async_mode
        self.focus_each_step = bool(focus_each_step)
        self.capture_margin = int(capture_margin)
        self.reuse_last_frame_on_capture_fail = bool(reuse_last_frame_on_capture_fail)
        self.capture_retry_delay_s = float(capture_retry_delay_s)
        self.hard_fail_after = int(hard_fail_after)
        self.hard_fail_sleep_s = float(hard_fail_sleep_s)
        self._consecutive_capture_failures = 0
        self._hard_fail_logged = False

        self.enable_controller = bool(enable_controller)
        self.gamepad_emulator = None
        if self.enable_controller:
            self.gamepad_emulator = GamepadEmulator(
                controller_type=controller_type,
                system=os_name,
                invert_y_axis=bool(invert_y_axis),
            )
        proc_info = get_process_info(game)

        self.game_pid = proc_info["pid"]
        self.game_arch = proc_info["architecture"]
        self.game_window_name = proc_info["window_name"]

        print(f"Game process found: {self.game} (PID: {self.game_pid}, Arch: {self.game_arch}, Window: {self.game_window_name})")

        if self.game_pid is None:
            raise Exception(f"Could not find PID for game: {game}")


        self.observation_space = Box(
            low=0, high=255, shape=(self.image_height, self.image_width, 3), dtype="uint8"
        )

        # Define a unified action space
        self.action_space = Dict(
            {
                "BACK": Discrete(2),
                "GUIDE": Discrete(2),
                "RIGHT_SHOULDER": Discrete(2),
                "RIGHT_TRIGGER": Box(low=0.0, high=1.0, shape=(1,)),
                "LEFT_TRIGGER": Box(low=0.0, high=1.0, shape=(1,)),
                "LEFT_SHOULDER": Discrete(2),
                "AXIS_RIGHTX": Box(low=-32768.0, high=32767, shape=(1,)),
                "AXIS_RIGHTY": Box(low=-32768.0, high=32767, shape=(1,)),
                "AXIS_LEFTX": Box(low=-32768.0, high=32767, shape=(1,)),
                "AXIS_LEFTY": Box(low=-32768.0, high=32767, shape=(1,)),
                "LEFT_THUMB": Discrete(2),
                "RIGHT_THUMB": Discrete(2),
                "DPAD_UP": Discrete(2),
                "DPAD_RIGHT": Discrete(2),
                "DPAD_DOWN": Discrete(2),
                "DPAD_LEFT": Discrete(2),
                "WEST": Discrete(2),
                "SOUTH": Discrete(2),
                "EAST": Discrete(2),
                "NORTH": Discrete(2),
                "START": Discrete(2),
            }
        )

        # Determine window name
        windows = pwc.getAllWindows()
        self.game_window = None
        for window in windows:
            if window.title == self.game_window_name:
                self.game_window = window
                break

        if not self.game_window:
            raise Exception(f"No window found with game name: {self.game}")

        self.game_hwnd = getattr(self.game_window, "_hWnd", None)

        self.game_window.activate()
        l, t, r, b = self.game_window.left, self.game_window.top, self.game_window.right, self.game_window.bottom
        m = max(0, self.capture_margin)
        w = max(1, (r - l) - 2 * m)
        h = max(1, (b - t) - 2 * m)
        self.bbox = (l + m, t + m, w, h)

        # Initialize speedhack client if using DLL injection
        self.speedhack_client = None

        # Get the screenshot backend
        if screenshot_backend == "dxcam":
            self.screenshot_backend = DxcamScreenshotBackend(
                self.bbox,
                reuse_last_on_fail=self.reuse_last_frame_on_capture_fail,
            )
        elif screenshot_backend == "pyautogui":
            self.screenshot_backend = PyautoguiScreenshotBackend(self.bbox)
        else:
            raise ValueError("Unsupported screenshot backend. Use 'dxcam' or 'pyautogui'.")


    def calculate_step_duration(self):
        """
        Calculate the step duration based on game speed and environment FPS.

        Returns:
        float: Calculated step duration.

        Example:
        If game_speed=1.0 and env_fps=10, then step_duration
        will be 0.1 seconds.
        """
        return 1.0 / (self.env_fps * self.game_speed)

    def pause(self):
        """
        Pause the game using the specified method.
        """
        if self.speedhack_client is not None:
            self.speedhack_client.set_speed(0.0)

    def unpause(self):
        """
        Unpause the game using the specified method.
        """
        if self.speedhack_client is not None:
            self.speedhack_client.set_speed(1.0)



    def perform_action(self, action, duration):
        """
        Perform the action without handling the game pause/unpause.

        Parameters:
        action (dict): Action to be performed.
        duration (float): Duration for the action step.
        """
        if self.focus_each_step:
            try:
                self.game_window.activate()
            except Exception:
                pass
            if self.game_hwnd is not None:
                try:
                    win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(self.game_hwnd)
                except Exception:
                    pass
        if self.gamepad_emulator is not None:
            self.gamepad_emulator.step(action)
        start = time.perf_counter()
        self.unpause()
        # Wait until the next step
        end = start + duration
        time.sleep(max(0.0, end - time.perf_counter()))
        self.pause()

    def step(self, action, step_duration=None):
        """
        Perform an action in the game environment and return the observation.

        Parameters:
        action (dict): Dictionary of the action to be performed. Keys are control names,
                    and values are their respective states.
        step_duration (float, optional): Duration for which the action should be performed.

        Returns:
        tuple: (obs, reward, terminated, truncated, info) where obs is the observation of the game environment after performing the action.
        """
        # Determine the duration for this step
        duration = step_duration if step_duration is not None else self.step_duration

        self.perform_action(action, duration)

        obs = self.render()  # Render after pausing the game

        reward = 0.0
        terminated = False
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state.

        Parameters:
        seed (int, optional): Random seed.
        options (dict, optional): Additional options for reset.
        """
        try:
            self.game_window.activate()
        except Exception:
            pass
        if getattr(self, "game_hwnd", None) is not None:
            try:
                win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                win32gui.SetForegroundWindow(self.game_hwnd)
            except Exception:
                pass
        if self.gamepad_emulator is not None:
            self.gamepad_emulator.wakeup(duration=0.1)
            time.sleep(1.0)

    def close(self):
        """
        Close the environment and release any resources.
        """
        pass  # Implement env close logic here

    def render(self):
        """
        Render the current state of the game window as an observation.

        Returns:
        Image: Observation of the game environment.
        """
        while True:
            screenshot = self.screenshot_backend.screenshot()
            if screenshot is not None:
                self._consecutive_capture_failures = 0
                self._hard_fail_logged = False
                break

            self._consecutive_capture_failures += 1
            if self._consecutive_capture_failures >= self.hard_fail_after and not self._hard_fail_logged:
                print(
                    "\n!!! DXCAM HARD-FAIL: capture repeatedly failing; pausing actions until capture recovers !!!\n"
                    "If this persists:\n"
                    "- Switch the game to borderless/windowed mode (exclusive fullscreen often breaks Desktop Duplication).\n"
                    "- Try `scripts/play.py --screenshot-backend pyautogui`.\n"
                    "- Try increasing `--capture-margin` (some games have invisible borders/overlays).\n"
                )
                self._hard_fail_logged = True

            print(f"DXCAM failed to capture frame ({self._consecutive_capture_failures}x); retrying...")
            time.sleep(self.capture_retry_delay_s)
            if self._consecutive_capture_failures >= self.hard_fail_after:
                time.sleep(self.hard_fail_sleep_s)

        screenshot = screenshot.resize((self.image_width, self.image_height))

        return screenshot
