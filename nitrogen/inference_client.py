import time
import pickle

import numpy as np
import zmq

class ModelClient:
    """Client for model inference server."""
    
    def __init__(self, host="localhost", port=5555, timeout_ms: int = 120_000):
        """
        Initialize client connection.
        
        Args:
            host: Server hostname or IP
            port: Server port
            timeout_ms: Receive timeout in milliseconds
        """
        self.host = host
        self.port = port
        self.timeout_ms = int(timeout_ms)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        
        print(f"Connected to model server at {host}:{port} (timeout={self.timeout_ms}ms)")
    
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
                response = pickle.loads(self.socket.recv())
                break

            if elapsed_s >= next_log_s:
                print(f"Waiting for model server reply... {elapsed_s:.1f}s")
                next_log_s += 5.0
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["pred"]
    
    def reset(self):
        """Reset the server's session (clear buffers)."""
        request = {"type": "reset"}
        
        self.socket.send(pickle.dumps(request))
        try:
            response = pickle.loads(self.socket.recv())
        except zmq.Again as e:
            raise TimeoutError(
                f"Timed out waiting for server reply after {self.timeout_ms}ms during reset."
            ) from e
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        print("Session reset")

    def info(self) -> dict:
        """Get session info from the server."""
        request = {"type": "info"}
        
        self.socket.send(pickle.dumps(request))
        try:
            response = pickle.loads(self.socket.recv())
        except zmq.Again as e:
            raise TimeoutError(
                f"Timed out waiting for server reply after {self.timeout_ms}ms during info()."
            ) from e
        
        if response["status"] != "ok":
            raise RuntimeError(f"Server error: {response.get('message', 'Unknown error')}")
        
        return response["info"]

    def close(self):
        """Close the connection."""
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()
        self.context.term()
        print("Connection closed")
    
    def __enter__(self):
        """Support for context manager."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection when exiting context."""
        self.close()
