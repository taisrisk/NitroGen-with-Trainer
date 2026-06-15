import threading
import time
import json
import numpy as np
import base64
from io import BytesIO
from PIL import Image

# Use the real helix-db SDK
try:
    from helix_db import client as hlx_client
    import helix_db.dsl as h
    HAS_HELIX = True
except ImportError:
    HAS_HELIX = False
    print("[Ollama] Warning: helix_db not found. Falling back to mock HelixDB.")

import ollama

class RealHelixDB:
    def __init__(self, db_url="http://127.0.0.1:8000"):
        self.db_url = db_url
        if HAS_HELIX:
            # HelixDB local Client.
            self.client = hlx_client.HelixDBClient(db_url)
        else:
            self.client = None
            self.store = {} # Fallback

    def write(self, key, value):
        if self.client:
            try:
                # Store node in HelixDB.
                import asyncio
                # The HelixDB SDK uses specific syntax for creating nodes and edges.
                # Assuming simple property storage:
                q = h.write_batch([
                    h.g.add_node("memory", key, {"data": h.stringify_json(value)})
                ])
                # Provide a fresh event loop since we are in a sub-thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.client.run(q))
                loop.close()
            except Exception as e:
                print(f"[HelixDB] Write Error: {e}")
        else:
            self.store[key] = value

    def get(self, key, default=None):
        if self.client:
            try:
                import asyncio
                q = h.read_batch([
                    h.g.V("memory", key)
                ])
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                res = loop.run_until_complete(self.client.run(q))
                loop.close()
                if res and res.results and len(res.results) > 0:
                    node = res.results[0]
                    return h.parse_json_structural(node.get("data", default))
                return default
            except Exception as e:
                print(f"[HelixDB] Get Error: {e}")
                return default
        else:
            return self.store.get(key, default)

class OllamaSystem2:
    def __init__(self, model_name="qwen2.5:3b", interval_s=1.0):
        # Qwen2.5 supports multi-modal images and text.
        self.model_name = model_name
        self.interval_s = interval_s
        self.db = RealHelixDB()
        self.current_strategy = "STRATEGY: IDLE"
        self._lock = threading.Lock()
        self.running = False
        self._thread = None

        self.game_state = {
            "hp": "100%",
            "status": "normal",
            "recent_events": [],
            "image": None # Holds the latest frame base64 string
        }

    def start(self):
        self.running = True
        print(f"[Ollama] Pre-warming model {self.model_name}...")
        try:
            # Pre-warm and keep alive for 12 hours so it never drops from VRAM
            ollama.generate(model=self.model_name, prompt="Hello", keep_alive="12h")
        except Exception as e:
            print(f"[Ollama] Pre-warm failed (is Ollama running?): {e}")

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Ollama] System 2 started. Model: {self.model_name}, Hz: {1.0/self.interval_s:.2f}")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def update_state(self, key, value):
        with self._lock:
            self.game_state[key] = value

    def set_image(self, pil_img):
        # Convert PIL image to base64 for Ollama Multimodal
        buffered = BytesIO()
        # Resize aggressively to save prompt processing time
        small_img = pil_img.resize((256, 256))
        small_img.save(buffered, format="JPEG", quality=70)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        with self._lock:
            self.game_state["image"] = img_str

    def push_event(self, event_str):
        with self._lock:
            self.game_state["recent_events"].append(event_str)
            if len(self.game_state["recent_events"]) > 5:
                self.game_state["recent_events"].pop(0)

    def get_strategy(self):
        with self._lock:
            return self.current_strategy

    def _loop(self):
        while self.running:
            start_time = time.time()
            try:
                self._evaluate_state()
            except Exception as e:
                pass

            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval_s - elapsed)
            time.sleep(sleep_time)

    def _evaluate_state(self):
        with self._lock:
            state_copy = self.game_state.copy()
            events = "\n".join(state_copy["recent_events"])

        prompt = (
            f"Game State:\nHP: {state_copy.get('hp')}\nStatus: {state_copy.get('status')}\n"
            f"Recent Events:\n{events}\n"
            "Output a single concise STRATEGY macro based on the image and text. Examples: 'STRATEGY: KITE_AND_HEAL', 'STRATEGY: AGGRESSIVE_CLOSE_QUARTERS', 'STRATEGY: DODGE'.\n"
            "Response:"
        )

        try:
            kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "12h", # Ensure model stays in memory
                "options": {
                    "num_ctx": 2048,
                    "temperature": 0.1
                }
            }
            if state_copy.get("image") is not None:
                kwargs["images"] = [state_copy["image"]]

            response = ollama.generate(**kwargs)
            new_strat = response.get("response", "").strip()
            if "STRATEGY:" in new_strat:
                with self._lock:
                    self.current_strategy = new_strat
                print(f"[Ollama] New strategy: {self.current_strategy}")

        except Exception as e:
            pass
