import threading
import time
import base64
import json
import re
from io import BytesIO
from PIL import Image

class MicroLLMBrain:
    """
    System 2: The slow, smart reasoning layer.

    HARD FIX 1: MICRO LLM ROLE
    The LLM is NOT controlling the mouse. It is the hypothesis generator.

    HARD FIX 2: INTENT PERSISTENCE
    The LLM cannot flip-flop every frame. It issues an Intent that lasts
    for `intent_lifetime` steps, preventing chaotic oscillation.
    """
    def __init__(self, memory_db, model_name="qwen2.5:3b", interval_s=2.0, intent_lifetime_s=3.0):
        self.memory_db = memory_db
        self.interval_s = interval_s
        self.model_name = model_name

        self.current_intent = "INTENT: OBSERVE"
        self.intent_expiration = time.time()
        self.intent_lifetime_s = intent_lifetime_s

        self._lock = threading.Lock()
        self.running = False
        self._thread = None

        self.shared_state = {
            "semantic_state": {},
            "status": "active"
        }

        try:
            import ollama
            self.ollama = ollama
            print(f"[Brain] Pre-warming {self.model_name}...")
            self.ollama.generate(model=self.model_name, prompt="init", keep_alive="12h")
        except ImportError:
            print("[Brain] Error: 'ollama' python package not installed.")
            self.ollama = None
        except Exception as e:
            print(f"[Brain] Warning: Failed to contact local Ollama: {e}")
            self.ollama = None

    def start(self):
        if not self.ollama:
            print("[Brain] Starting in DUMMY mode (Ollama offline/missing).")

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[Brain] Hypothesis generator loop started at {1.0/self.interval_s:.2f}Hz")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def update_state(self, state_dict: dict):
        with self._lock:
            self.shared_state.update(state_dict)

    def get_current_intent(self) -> str:
        with self._lock:
            if time.time() > self.intent_expiration:
                return "INTENT: SAFE_BASELINE"
            return self.current_intent

    def _loop(self):
        while self.running:
            start_time = time.time()
            try:
                with self._lock:
                    semantic_state = self.shared_state.get("semantic_state", {})
                    danger = semantic_state.get("danger", "low")
                    time_to_expire = self.intent_expiration - time.time()

                if time_to_expire < 0.5 or danger == "critical":
                    self._think()
            except Exception as e:
                pass

            elapsed = time.time() - start_time
            sleep_time = max(0, self.interval_s - elapsed)
            time.sleep(sleep_time)

    def _think(self):
        with self._lock:
            status = self.shared_state.get("status", "unknown")
            semantic_state = self.shared_state.get("semantic_state", {})

        boss_action = semantic_state.get("boss_action", "unknown")
        state_hash = f"state_boss_{boss_action}"

        memory_context = self.memory_db.query_relevant_experience(state_hash)

        state_json = json.dumps(semantic_state, indent=2)

        prompt = (
            f"Current Logical State:\n{state_json}\n\n"
            "You are the hypothesis generator. Decide the next macro strategy.\n"
            "Output a short INTENT macro.\n"
            "Examples: 'INTENT: WAIT_FOR_PUNISH', 'INTENT: ATTACK', 'INTENT: DODGE'.\n"
        )
        if memory_context:
            prompt = f"EPISODIC MEMORY (Avoid past mistakes!):\n{memory_context}\n\n" + prompt

        prompt += "Response:"

        if self.ollama:
            kwargs = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "12h",
                "options": {"num_ctx": 2048, "temperature": 0.1}
            }

            try:
                response = self.ollama.generate(**kwargs)
                raw_text = response.get("response", "")

                match = re.search(r'(INTENT:\s*[A-Z_]+)', raw_text)
                if match:
                    clean_intent = match.group(1).strip()
                    with self._lock:
                        if self.current_intent != clean_intent:
                            self.current_intent = clean_intent
                            self.intent_expiration = time.time() + self.intent_lifetime_s
                            print(f"[Brain] Shifted intent -> {self.current_intent} (Valid for {self.intent_lifetime_s}s)")
            except Exception as e:
                pass
