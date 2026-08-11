"""
plugins/base.py — Plugin Interface
====================================
All plugins must inherit from DoorbellPlugin and implement
whichever hooks they need. Unimplemented hooks are no-ops.
"""

from abc import ABC
from dataclasses import dataclass, field
from typing import Optional
import logging


@dataclass
class VisitorEvent:
    face_index: int = -1
    name: str = "Unknown"
    is_known: bool = False
    intent: str = "unknown"
    intent_confidence: float = 0.0
    spoken_name: str = ""
    transcript: str = ""
    face_crop_path: str = ""
    timestamp: str = ""
    language: str = "ur"
    conflict: bool = False
    extra: dict = field(default_factory=dict)


class DoorbellPlugin(ABC):
    name: str = "unnamed_plugin"
    version: str = "1.0.0"
    enabled: bool = True

    def __init__(self):
        self.logger = logging.getLogger(f"plugin.{self.name}")

    def on_load(self) -> None:
        pass

    def on_unload(self) -> None:
        pass

    def on_face_detected(self, frame, face, is_real: bool) -> None:
        pass

    def on_visitor_identified(self, event: VisitorEvent) -> None:
        pass

    def on_dialogue_start(self, event: VisitorEvent) -> None:
        pass

    def on_dialogue_complete(self, event: VisitorEvent) -> None:
        pass

    def on_cooldown_start(self, face_index: int) -> None:
        pass

    def on_system_startup(self) -> None:
        pass