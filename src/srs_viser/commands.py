"""Preserve control order while replacing consecutive samples of one drag."""

from collections import deque
from threading import Lock


class CommandQueue:
    """Callbacks share this queue; the simulation thread drains it once per frame.

    Only adjacent samples of the same continuous control can replace one another.
    A different control or a reset/mode/play action is an ordering boundary.
    Coalescing on insertion also keeps a single-control drag from growing the queue.
    """

    _continuous = frozenset(
        (
            "angle",
            "joint",
            "target",
            "amplitude",
            "frequency",
            "mesh_toggle",
            "path_radius",
            "max_joint_speed",
            "manifold_samples",
        )
    )

    def __init__(self):
        self._pending = deque()
        self._lock = Lock()

    @staticmethod
    def _key(command):
        name, value = command
        return (name, value[0]) if name == "joint" else name

    def put(self, command):
        with self._lock:
            if (
                command[0] in self._continuous
                and self._pending
                and self._key(self._pending[-1]) == self._key(command)
            ):
                self._pending[-1] = command
            else:
                self._pending.append(command)

    def drain(self):
        with self._lock:
            commands = list(self._pending)
            self._pending.clear()
        return commands
