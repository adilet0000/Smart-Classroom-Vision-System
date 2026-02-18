from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import os
import yaml
from dotenv import load_dotenv


def _deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
   result = dict(base)
   for k, v in extra.items():
      if isinstance(v, dict) and isinstance(result.get(k), dict):
         result[k] = _deep_merge(result[k], v)
      else:
         result[k] = v
   return result


@dataclass(frozen=True)
class AppConfig:
   window_name: str = "Engagement Detection"
   show_fps: bool = True


@dataclass(frozen=True)
class VideoConfig:
   source: Any = 0
   width: int = 1280
   height: int = 720
   target_fps: int = 30
   mirror: bool = True


@dataclass(frozen=True)
class DebugConfig:
   draw_center_dot: bool = True


@dataclass(frozen=True)
class Config:
   app: AppConfig
   video: VideoConfig
   debug: DebugConfig


def load_config(config_path: str = "configs/default.yaml") -> Config:
   load_dotenv(override=False)

   path = Path(config_path)
   data: dict[str, Any] = {}
   if path.exists():
      data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

   # override из env (если захочешь)
   env_overrides: dict[str, Any] = {}
   video_source = os.getenv("VIDEO_SOURCE")
   if video_source:
      env_overrides = {"video": {"source": video_source}}

   merged = _deep_merge(data, env_overrides)

   app = merged.get("app", {})
   video = merged.get("video", {})
   debug = merged.get("debug", {})

   return Config(
      app=AppConfig(
         window_name=str(app.get("window_name", "Engagement Detection")),
         show_fps=bool(app.get("show_fps", True)),
      ),
      video=VideoConfig(
         source=video.get("source", 0),
         width=int(video.get("width", 1280)),
         height=int(video.get("height", 720)),
         target_fps=int(video.get("target_fps", 30)),
         mirror=bool(video.get("mirror", True)),
      ),
      debug=DebugConfig(
         draw_center_dot=bool(debug.get("draw_center_dot", True)),
      ),
   )
