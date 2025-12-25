import time
import pickle

import numpy as np
import zmq

class ModelClient:
    """Client for model inference server."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        timeout_ms: int = 120_000,
        reconnect_delay_s: float = 0.5,
    ):
        """
        Initialize client connection.
        
        Args:
            host: Server hostname or IP
            port: Server port
            timeout_ms: Receive timeout in milliseconds
            reconnect_delay_s: Backoff delay after reconnect attempts
        """
        self.host = host
        self.port = port
        self.timeout_ms = int(timeout_ms)
        self.reconnect_delay_s = float(reconnect_delay_s)

        self.context = zmq.Context()
        self.socket = None
        self._connect()
        
        print(f"Connected to model server at {host}:{port} (timeout={self.timeout_ms}ms)")

    def _connect(self):
        if self.socket is not None:
            return
        sock = self.context.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.connect(f"tcp://{self.host}:{self.port}")
        self.socket = sock

    def _close_socket(self):
        if self.socket is None:
            return
        try:
            self.socket.setsockopt(zmq.LINGER, 0)
        except Exception:
            pass
        try:
            self.socket.close()
        finally:
            self.socket = None

    def reconnect(self):
        """Force a new REQ socket (resets REQ/REP state after timeouts or server restarts)."""
        self._close_socket()
        self._connect()

    def wait_for_server(self, timeout_s: float = 60.0, retry_delay_s: float | None = None):
        """Block until the server responds to an info() request (or timeout)."""
        retry_delay_s = self.reconnect_delay_s if retry_delay_s is None else float(retry_delay_s)
        deadline = time.perf_counter() + float(timeout_s)
        last_log = 0.0
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"Timed out waiting for server after {timeout_s}s")
            try:
                self.info()
                return
            except Exception:
                self.reconnect()
                now = time.perf_counter()
                if now - last_log >= 5.0:
                    print("Waiting for model server to become ready...")
                    last_log = now
                time.sleep(min(retry_delay_s, max(0.05, remaining)))
    
    def _request(self, request: dict) -> dict:
        self._connect()
        assert self.socket is not None

        self.socket.send(pickle.dumps(request))
        t0 = time.perf_counter()
        next_log_s = 5.0
        while True:
            elapsed_s = time.perf_counter() - t0
            remaining_ms = self.timeout_ms - int(elapsed_s * 1000)
            if remaining_ms <= 0:
                raise TimeoutError(
                    f"Timed out waiting for server reply after {self.timeout_ms}ms. "
                    f"If the server is still computing, increase scripts/play.py --timeout-ms."
                )

            poll_ms = min(1000, remaining_ms)
            if self.socket.poll(timeout=poll_ms, flags=zmq.POLLIN):
                return pickle.loads(self.socket.recv())

            if elapsed_s >= next_log_s and request.get("type") == "predict":
                print(f"Waiting for model server reply... {elapsed_s:.1f}s")
                next_log_s += 5.0

    def predict(self, image: np.ndarray) -> dict:
        """
        Send an image and receive predicted actions.
        
        Args:
            image: numpy array (H, W, 3) in RGB format
            
        Returns:
            List of action dicts, each containing:
                - j_left: [x, y] left joystick position
                - j_right: [x, y] right joystick position  
                - buttons: list of button values
        """
        request = {
            "type": "predict",
            "image": image
        }
        try:
            response = self._request(request)
        except Exception:
            # After a failed/timeout request, a REQ socket is no longer safe to reuse.
            self.reconnect()
            raise
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["pred"]
    
    def reset(self):
        """Reset the server's session (clear buffers)."""
        request = {"type": "reset"}
        try:
            response = self._request(request)
        except Exception:
            self.reconnect()
            raise
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        print("Session reset")

    def info(self) -> dict:
        """Get session info from the server."""
        request = {"type": "info"}
        try:
            response = self._request(request)
        except Exception:
            self.reconnect()
            raise
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["info"]

    def close(self):
        """Close the connection."""
        self._close_socket()
        self.context.term()
        print("Connection closed")
    
    def __enter__(self):
        """Support for context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context."""
        self.close()
