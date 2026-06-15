import queue
import threading
import time

class AgentPipeline:
    """
    4-Thread Queue Architecture for ultra-low latency execution.
    Isolates blocking IO (screen capture, model inference, hardware execution)
    so the agent runs at maximum frames per second.
    """
    def __init__(self, capture_fn, vision_fn, policy_fn, execution_fn):
        self.capture_fn = capture_fn
        self.vision_fn = vision_fn
        self.policy_fn = policy_fn
        self.execution_fn = execution_fn

        # N-1 Queue implementation (maxsize=1) guarantees we only ever process the absolute
        # freshest frame. If the pipeline lags, old frames are immediately dropped
        # instead of creating a latency buffer.
        self.raw_frame_q = queue.Queue(maxsize=1)
        self.state_q = queue.Queue(maxsize=1)
        self.action_q = queue.Queue(maxsize=1)

        self.running = False

        # Shared Intent (updated by asynchronous System 2 Brain)
        self.current_intent = "WAIT"
        self._intent_lock = threading.Lock()

    def set_intent(self, intent: str):
        with self._intent_lock:
            self.current_intent = intent

    def get_intent(self) -> str:
        with self._intent_lock:
            return self.current_intent

    def _thread_capture(self):
        while self.running:
            try:
                frame = self.capture_fn()
                # Overwrite oldest to minimize latency
                if self.raw_frame_q.full():
                    try:
                        self.raw_frame_q.get_nowait()
                    except queue.Empty:
                        pass
                self.raw_frame_q.put(frame)
            except Exception as e:
                print(f"[Capture Error] {e}")

    def _thread_vision(self):
        while self.running:
            try:
                frame = self.raw_frame_q.get(timeout=0.1)
                state = self.vision_fn(frame)

                if self.state_q.full():
                    try:
                        self.state_q.get_nowait()
                    except queue.Empty:
                        pass
                self.state_q.put(state)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Vision Error] {e}")

    def _thread_policy(self):
        while self.running:
            try:
                state = self.state_q.get(timeout=0.1)
                intent = self.get_intent()
                action = self.policy_fn(state, intent)

                if self.action_q.full():
                    try:
                        self.action_q.get_nowait()
                    except queue.Empty:
                        pass
                self.action_q.put(action)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Policy Error] {e}")

    def _thread_execute(self):
        while self.running:
            try:
                action = self.action_q.get(timeout=0.1)
                self.execution_fn(action)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Execution Error] {e}")

    def start(self):
        self.running = True
        self.threads = [
            threading.Thread(target=self._thread_capture, daemon=True, name="Pipeline-Capture"),
            threading.Thread(target=self._thread_vision, daemon=True, name="Pipeline-Vision"),
            threading.Thread(target=self._thread_policy, daemon=True, name="Pipeline-Policy"),
            threading.Thread(target=self._thread_execute, daemon=True, name="Pipeline-Execution"),
        ]
        for t in self.threads:
            t.start()
        print("[Pipeline] 4-Thread Low-Latency Engine Started.")

    def stop(self):
        self.running = False
        for t in self.threads:
            t.join(timeout=2.0)
        print("[Pipeline] Engine Stopped.")
