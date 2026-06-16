import time
import argparse
import yaml
import cv2
import numpy as np
from PIL import Image

from nitrogen.game_env import GamepadEnv
try:
    from nitrogen.kbm_emulator import KeyboardMouseEmulator, KBMConfig
except ImportError:
    KeyboardMouseEmulator = None

from core.vision.state_encoder import VisionStateEncoderFactory
from core.memory.helix_memory import EpisodicMemory
from core.brain.system2 import MicroLLMBrain
from core.policy.system1 import FastPolicyExecutor
from core.pipeline import AgentPipeline

def load_config(config_path="config/system.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Failed to load config: {e}. Using defaults.")
        return {
            "agent": {"target_fps": 30, "resolution": [1280, 720]},
            "brain": {"interval_s": 1.0, "model_name": "qwen2.5:3b"},
            "memory": {"helixdb_url": "http://127.0.0.1:8000"}
        }

def run_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process", type=str, default="celeste.exe", help="Game/App to hook")
    parser.add_argument("--ckpt", type=str, help="Path to actual NitroGen policy weights")
    args = parser.parse_args()

    cfg = load_config()
    print("\n--- Booting Universal Cognitive Architecture ---")

    # 1. Initialize Memory (HelixDB)
    memory_db = EpisodicMemory(db_url=cfg["memory"]["helixdb_url"])

    # 2. Initialize Brain (Ollama/Qwen asynchronous reasoning loop)
    brain = MicroLLMBrain(
        memory_db=memory_db,
        model_name=cfg["brain"]["model_name"],
        interval_s=1.0 / cfg["brain"].get("reasoning_hz", 1.0),
        intent_lifetime_s=3.0
    )
    brain.start()

    # 3. Initialize Vision (Semantic Temporal State Extractor)
    vision = VisionStateEncoderFactory.create(
        backend="siglip",
        target_resolution=tuple(cfg["agent"].get("resolution", [256, 256]))
    )

    # 4. Initialize Policy (Fast Hierarchical Action Head)
    policy = FastPolicyExecutor(ckpt_path=args.ckpt)

    # 5. Environment Hook
    env = GamepadEnv(
        game=args.process,
        game_speed=1.0,
        env_fps=cfg["agent"]["target_fps"],
        async_mode=True,
        image_width=cfg["agent"]["resolution"][0],
        image_height=cfg["agent"]["resolution"][1],
        enable_controller=False, # Using KBM
        screenshot_backend="dxcam"
    )
    env.reset()
    env.pause()

    if KeyboardMouseEmulator:
        kbm = KeyboardMouseEmulator(token_to_keys={}, config=KBMConfig())
    else:
        kbm = None

    # Pipeline Interface Definitions
    def capture_fn():
        # Actual environment screen capture
        obs = env.render()
        if isinstance(obs, Image.Image):
            return np.array(obs)
        return obs

    def execute_fn(hierarchical_action):
        hardware_action = policy.translate_to_hardware(hierarchical_action)
        if kbm:
            kbm.step_keys(
                keys_down=hardware_action["keys_down"],
                lmb=hardware_action["lmb"],
                rmb=hardware_action["rmb"],
                mouse_dx=hardware_action["mouse_dx"],
                mouse_dy=hardware_action["mouse_dy"],
                duration_s=0.0
            )

    # 6. Initialize Multi-Threaded Latency Pipeline
    pipeline = AgentPipeline(
        capture_fn=capture_fn,
        vision_fn=vision.encode,
        policy_fn=policy.decide_hierarchical_action,
        execution_fn=execute_fn
    )

    print(f"\n--- Pipeline Online. Target FPS: {cfg['agent']['target_fps']} ---")
    pipeline.start()

    try:
        while True:
            # Main thread manages the slow LLM intent sync and memory logging
            # extracting state from the pipeline queues safely

            # (In a full implementation, we'd hook specific event flags from the
            # vision encoder to trigger memory_db.log_experience here).

            # Sync the latest intent into the high-speed pipeline
            current_intent = brain.get_current_intent()
            pipeline.set_intent(current_intent)

            time.sleep(0.1) # Main thread spins slowly while pipeline goes fast

    except KeyboardInterrupt:
        print("\nShutting down Agent...")
    finally:
        pipeline.stop()
        brain.stop()
        if kbm:
            kbm.release_all()
        env.unpause()
        env.close()

if __name__ == "__main__":
    run_agent()
