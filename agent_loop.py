import time
import argparse
import yaml

from nitrogen.game_env import GamepadEnv
try:
    from nitrogen.kbm_emulator import KeyboardMouseEmulator, KBMConfig
except ImportError:
    KeyboardMouseEmulator = None

from core.vision.state_encoder import VisionStateEncoder
from core.memory.helix_memory import EpisodicMemory
from core.brain.system2 import MicroLLMBrain
from core.policy.system1 import FastPolicyExecutor

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
    # The Brain evaluates at a slow Hz, maintaining intent persistence
    brain = MicroLLMBrain(
        memory_db=memory_db,
        model_name=cfg["brain"]["model_name"],
        interval_s=1.0 / cfg["brain"].get("reasoning_hz", 1.0),
        intent_lifetime_s=3.0
    )
    brain.start()

    # 3. Initialize Vision (Semantic Temporal State Extractor)
    vision = VisionStateEncoder(target_resolution=tuple(cfg["agent"].get("resolution", [256, 256])))

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

    if KeyboardMouseEmulator:
        kbm = KeyboardMouseEmulator(token_to_keys={}, config=KBMConfig())
    else:
        kbm = None

    print(f"\n--- System Online. Target FPS: {cfg['agent']['target_fps']} ---")

    env.reset()
    env.pause()
    obs = env.render()

    step_duration = 1.0 / cfg['agent']['target_fps']
    step_count = 0

    # Track the last action taken for memory logging
    last_action_macro = "WAIT"

    try:
        import numpy as np
        from PIL import Image

        while True:
            loop_start = time.perf_counter()

            # --- 1. OBSERVE (Vision) ---
            if isinstance(obs, Image.Image):
                raw_pixels = np.array(obs)
            else:
                raw_pixels = obs

            state_dict = vision.encode(raw_pixels)

            # --- 2. UNDERSTAND (Brain Sync) ---
            brain.update_state({
                "semantic_state": state_dict["semantic_state"],
                "status": state_dict["status"]
            })

            # The intent persists across multiple frames to prevent chaotic oscillation
            current_intent = brain.get_current_intent()

            # --- 3. DECIDE (Policy) ---
            # Fast neural evaluation: Semantic State + Intent -> Hierarchical Action
            hierarchical_action = policy.decide_hierarchical_action(state_dict["semantic_state"], current_intent)

            # --- 4. ACT (Execution) ---
            # Action translates directly to KBM outputs regardless of LLM status
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

            obs = env.render()

            # --- 5. LOG & LEARN (Memory Loop) ---
            # Credit Assignment via Critical Factor tagging
            boss_action = state_dict["semantic_state"].get("boss_action", "unknown")
            state_hash = f"state_boss_{boss_action}"

            # Simulated death condition evaluation
            if hierarchical_action == "ATTACK_MELEE" and boss_action == "windup":
                memory_db.log_experience(
                    state_hash=state_hash,
                    intent=current_intent,
                    action=hierarchical_action,
                    result="died",
                    reward=-1,
                    critical_factor="unsafe healing window; boss was winding up"
                )

            last_action_macro = hierarchical_action

            step_count += 1
            if step_count % 100 == 0:
                print(f"[Loop {step_count}] Intent: {current_intent} | Policy Act: {hierarchical_action}")

            elapsed = time.perf_counter() - loop_start
            time.sleep(max(0, step_duration - elapsed))

    except KeyboardInterrupt:
        print("\nShutting down Agent...")
    finally:
        brain.stop()
        if kbm:
            kbm.release_all()
        env.unpause()
        env.close()

if __name__ == "__main__":
    run_agent()
