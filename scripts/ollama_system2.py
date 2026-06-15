import threading
import time
import requests
import json
import numpy as np

class HelixDB:
    """
    Ultra-lightweight episodic memory store.
    Simulates HelixDB or any fast Key/Value and vector database.
    """
    def __init__(self):
        self.store = {}

    def write(self, key, value):
        self.store[key] = value

    def get(self, key, default=None):
        return self.store.get(key, default)

class OllamaSystem2:
    def __init__(self, model_name="llama3.2:1b", interval_s=1.0):
        self.model_name = model_name
        self.interval_s = interval_s
        self.db = HelixDB()
        self.current_strategy = "STRATEGY: IDLE"
        self._lock = threading.Lock()
        self.running = False
        self._thread = None

        # We share state via this object
        self.game_state = {
            "hp": "100%",
            "status": "normal",
            "recent_events": []
        }

    def start(self):
        self.running = True
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
            try:
                self._evaluate_state()
            except Exception as e:
                print(f"[Ollama] Error: {e}")
            time.sleep(self.interval_s)

    def _evaluate_state(self):
        with self._lock:
            state_copy = self.game_state.copy()
            events = "\n".join(state_copy["recent_events"])
            prompt = (
                f"Game State:\nHP: {state_copy.get('hp')}\nStatus: {state_copy.get('status')}\n"
                f"Recent Events:\n{events}\n"
                "Output a single concise STRATEGY macro. Examples: 'STRATEGY: KITE_AND_HEAL', 'STRATEGY: AGGRESSIVE_CLOSE_QUARTERS', 'STRATEGY: DODGE'.\n"
                "Response:"
            )

        # Check if Ollama is running. If not, fallback gracefully.
        try:
            resp = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_ctx": 2048, "temperature": 0.1}
                },
                timeout=1.0
            )
            if resp.status_code == 200:
                result = resp.json()
                new_strat = result.get("response", "").strip()
                if "STRATEGY:" in new_strat:
                    with self._lock:
                        self.current_strategy = new_strat
                    print(f"[Ollama] New strategy: {self.current_strategy}")
        except requests.exceptions.RequestException:
            # If Ollama is offline, we just don't update
            pass
