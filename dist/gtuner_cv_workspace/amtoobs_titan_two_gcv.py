from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


PACKET_SIZE = 41
PACKET_MAGIC_MASK = 0xF0
PACKET_MAGIC_VALUE = 0xA0
SEQUENCE_OFFSET = 9
HOLD_MS_OFFSET = 29


def _read_int32_big_endian(payload: bytes, offset: int) -> int:
    return int.from_bytes(payload[offset:offset + 4], byteorder="big", signed=True)


def _write_int32_big_endian(payload: bytearray, offset: int, value: int) -> None:
    payload[offset:offset + 4] = int(value).to_bytes(4, byteorder="big", signed=True)


def _build_neutral_packet(sequence: int = 0) -> bytes:
    payload = bytearray(PACKET_SIZE)
    payload[0] = PACKET_MAGIC_VALUE
    _write_int32_big_endian(payload, SEQUENCE_OFFSET, sequence)
    _write_int32_big_endian(payload, HOLD_MS_OFFSET, 0)
    return bytes(payload)


class GCVWorker:
    def __init__(self, width: int, height: int):
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.config_path = Path(__file__).with_suffix(".json")
        self.config = self._load_config()
        self.packet_path = Path(self.config["packet_path"])
        self.stale_timeout_ms = int(self.config.get("stale_timeout_ms", 32))
        self.preview_width = max(1, int(self.config.get("preview_width", self.width or 1280)))
        self.preview_height = max(1, int(self.config.get("preview_height", self.height or 720)))
        self.drop_input_frame = bool(self.config.get("drop_input_frame", True))
        stats_path = self.config.get("stats_path", "")
        self.stats_path = Path(stats_path) if stats_path else None
        self.stats_interval_ms = max(100, int(self.config.get("stats_interval_ms", 1000)))
        self.read_attempts = max(1, int(self.config.get("read_attempts", 4)))
        self._last_sequence = 0
        self._empty_frame = np.zeros((self.preview_height, self.preview_width, 3), dtype=np.uint8)
        self._frame_count = 0
        self._stats_frame_count = 0
        self._last_packet_age_ms = None
        self._last_packet_seen_ns = None
        now_ns = time.perf_counter_ns()
        self._last_stats_ns = now_ns
        self._next_stats_ns = now_ns + self.stats_interval_ms * 1_000_000

    def __del__(self):
        pass

    def process(self, frame):
        self._frame_count += 1
        self._stats_frame_count += 1
        normalized = self._normalize_frame(frame)
        payload = self._read_feedback_packet()
        self._maybe_write_stats()
        return normalized, payload

    def _load_config(self) -> dict:
        script_path = Path(__file__).resolve()
        default_packet_path = script_path.with_name("titan_two_gcv.bin")
        default_stats_path = script_path.with_name("amtoobs_titan_two_gcv_stats.json")
        config = {
            "packet_path": str(default_packet_path),
            "stale_timeout_ms": 32,
            "preview_width": 16,
            "preview_height": 16,
            "drop_input_frame": True,
            "stats_path": str(default_stats_path),
            "stats_interval_ms": 1000,
        }
        if not self.config_path.exists():
            return config

        try:
            loaded = json.loads(self.config_path.read_text(encoding="utf-8-sig"))
        except Exception:
            return config

        if isinstance(loaded, dict):
            config.update(loaded)
        return config

    def _normalize_frame(self, frame):
        if frame is not None and not self.drop_input_frame:
            return frame
        return self._empty_frame

    def _maybe_write_stats(self) -> None:
        if self.stats_path is None:
            return

        now_ns = time.perf_counter_ns()
        if now_ns < self._next_stats_ns:
            return

        elapsed_ms = max(1.0, (now_ns - self._last_stats_ns) / 1_000_000.0)
        fps = self._stats_frame_count * 1000.0 / elapsed_ms
        stats = {
            "cv_fps": round(fps, 2),
            "frames": self._frame_count,
            "source_width": self.width,
            "source_height": self.height,
            "preview_width": self.preview_width,
            "preview_height": self.preview_height,
            "drop_input_frame": self.drop_input_frame,
            "read_attempts": self.read_attempts,
            "last_sequence": self._last_sequence,
            "last_packet_age_ms": self._last_packet_age_ms,
            "packet_path": str(self.packet_path),
        }

        try:
            self.stats_path.parent.mkdir(parents=True, exist_ok=True)
            self.stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        self._stats_frame_count = 0
        self._last_stats_ns = now_ns
        self._next_stats_ns = now_ns + self.stats_interval_ms * 1_000_000

    def _read_feedback_packet(self) -> bytes:
        stable = self._try_read_stable_packet()
        if stable is None:
            return _build_neutral_packet(self._last_sequence)

        flags = stable[0]
        if (flags & PACKET_MAGIC_MASK) != PACKET_MAGIC_VALUE:
            return _build_neutral_packet(self._last_sequence)

        now_ns = time.perf_counter_ns()
        try:
            sequence = _read_int32_big_endian(stable, SEQUENCE_OFFSET)
        except Exception:
            sequence = self._last_sequence

        if self._last_packet_seen_ns is None or sequence != self._last_sequence:
            self._last_packet_seen_ns = now_ns
            self._last_packet_age_ms = 0.0
            self._last_sequence = sequence
            return stable

        age_ms = (now_ns - self._last_packet_seen_ns) / 1_000_000.0
        self._last_packet_age_ms = round(age_ms, 2)
        if self.stale_timeout_ms > 0 and age_ms > self.stale_timeout_ms:
            return _build_neutral_packet(self._last_sequence)

        return stable

    def _is_valid_packet(self, payload: bytes) -> bool:
        return len(payload) == PACKET_SIZE and (payload[0] & PACKET_MAGIC_MASK) == PACKET_MAGIC_VALUE

    def _try_read_stable_packet(self) -> bytes | None:
        if not self.packet_path.exists():
            self._last_packet_age_ms = None
            return None

        best_packet = None
        best_sequence = None
        previous = None

        for _ in range(self.read_attempts):
            try:
                current = self.packet_path.read_bytes()
            except Exception:
                continue

            if not self._is_valid_packet(current):
                continue

            if previous == current:
                return current

            previous = current
            try:
                sequence = _read_int32_big_endian(current, SEQUENCE_OFFSET)
            except Exception:
                sequence = None

            if best_packet is None or sequence is None or best_sequence is None or sequence >= best_sequence:
                best_packet = current
                best_sequence = sequence

        return best_packet
