import pickle
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


class FileBackedSaver(InMemorySaver):
    def __init__(self, path: str) -> None:
        super().__init__()
        self.path = Path(path)
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        data = self.path.read_bytes()
        payload: dict[str, Any] = pickle.loads(data)
        storage = payload.get("storage", {})
        writes = payload.get("writes", {})
        blobs = payload.get("blobs", {})
        self.storage.clear()
        self.writes.clear()
        self.blobs.clear()
        for thread_id, namespaces in storage.items():
            for checkpoint_ns, checkpoints in namespaces.items():
                self.storage[thread_id][checkpoint_ns] = dict(checkpoints)
        for key, value in writes.items():
            self.writes[key] = dict(value)
        self.blobs.update(blobs)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "storage": {
                thread_id: {
                    checkpoint_ns: dict(checkpoints)
                    for checkpoint_ns, checkpoints in namespaces.items()
                }
                for thread_id, namespaces in self.storage.items()
            },
            "writes": {key: dict(value) for key, value in self.writes.items()},
            "blobs": dict(self.blobs),
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_bytes(pickle.dumps(payload))
        tmp.replace(self.path)

    def put(self, config, checkpoint, metadata, new_versions):
        result = super().put(config, checkpoint, metadata, new_versions)
        self._persist()
        return result

    def put_writes(self, config, writes, task_id, task_path: str = "") -> None:
        super().put_writes(config, writes, task_id, task_path=task_path)
        self._persist()

    def delete_thread(self, thread_id: str) -> None:
        super().delete_thread(thread_id)
        self._persist()
