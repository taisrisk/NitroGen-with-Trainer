import json

try:
    from helix_db import HelixDBClient
    from helix_db import DynamicQueryRequest
    from helix_db.dsl import write_batch, read_batch, stringify_json, parse_json_structural, g
    HAS_HELIX = True
except ImportError:
    HAS_HELIX = False

class EpisodicMemory:
    """
    HelixDB wrapper. Acts as the Long-term Brain.

    HARD FIX: FAILURE TAGGING & CREDIT ASSIGNMENT
    Stores experiences with explicit FAILURE REASONS to prevent 1000+ retries.
    """
    def __init__(self, db_url="http://127.0.0.1:8000"):
        self.db_url = db_url
        if HAS_HELIX:
            try:
                self.client = HelixDBClient(base_url=self.db_url)
            except Exception as e:
                print(f"[Memory] HelixDB Connection Warning: {e}")
                self.client = None
        else:
            self.client = None

        self.local_store = {}

    def log_experience(self, state_hash: str, intent: str, action: str, result: str, reward: int, critical_factor: str = None):
        if reward == 0 and result not in ["died", "boss_killed"]:
            return

        experience = {
            "intent": intent,
            "action": action,
            "result": result,
            "reward": reward,
            "critical_factor": critical_factor
        }

        existing = self._get_raw(state_hash) or []
        existing.append(experience)

        self._write_raw(state_hash, existing)
        print(f"[Memory] Tagged Experience Logged! State: '{state_hash}' | Factor: {critical_factor}")

    def query_relevant_experience(self, state_hash: str) -> str:
        experiences = self._get_raw(state_hash)
        if not experiences:
            return ""

        lines = []
        for exp in experiences[-5:]:
            factor_str = f" | Critical Factor: {exp['critical_factor']}" if exp.get('critical_factor') else ""
            lines.append(f"Past attempt: Action={exp['action']} -> Result={exp['result']}{factor_str}")

        return "\n".join(lines)

    def _write_raw(self, key: str, value: list):
        if self.client is not None:
            try:
                query = write_batch([
                    g.add_node("memory", key, {"data": stringify_json(value)})
                ])
                req = DynamicQueryRequest.write(query)
                self.client.execute(req)
            except Exception:
                self.local_store[key] = value
        else:
            self.local_store[key] = value

    def _get_raw(self, key: str):
        if self.client is not None:
            try:
                query = read_batch([
                    g.V("memory", key)
                ])
                req = DynamicQueryRequest.read(query)
                res = self.client.execute(req)

                if hasattr(res, 'results') and len(res.results) > 0:
                    node = res.results[0]
                    if isinstance(node, dict) and "data" in node:
                        return parse_json_structural(node["data"])
                return []
            except Exception:
                return self.local_store.get(key, [])
        else:
            return self.local_store.get(key, [])
