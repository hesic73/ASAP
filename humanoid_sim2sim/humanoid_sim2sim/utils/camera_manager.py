import mujoco
import numpy as np
from pathlib import Path

import imageio

from loguru import logger
from omegaconf import DictConfig


class CameraManager:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, video_cfg: DictConfig):
        self.video_cfg = video_cfg
        self.renderer = None
        self.camera = None
        self.frames = []
        self.tracking_body_name = video_cfg.camera_settings.get(
            "lookat_body")  # Ensure this exists in config

        if self.video_cfg.enabled:
            self.renderer = mujoco.Renderer(
                model, height=self.video_cfg.height, width=self.video_cfg.width)
            self.camera = mujoco.MjvCamera()
            self._setup_camera_properties(model, data)

    def _setup_camera_properties(self, model: mujoco.MjModel, data: mujoco.MjData):
        if not self.camera:
            return

        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE  # Default

        if self.tracking_body_name:
            tracking_body_id = model.body(self.tracking_body_name).id
            if tracking_body_id != -1:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.camera.trackbodyid = tracking_body_id
                self.camera.lookat = data.body(tracking_body_id).xpos.copy()
                logger.info(
                    f"Camera tracking body: '{self.tracking_body_name}' (ID: {tracking_body_id})")
            else:
                logger.warning(
                    f"Camera tracking body '{self.tracking_body_name}' not found. Using free camera.")
        else:
            logger.info(
                "No 'lookat_body' specified for camera. Using default free camera view.")
            # Sensible defaults for free camera if no tracking body
            # Look at a reasonable height around origin
            self.camera.lookat = np.array([0.0, 0.0, 0.75])

        self.camera.distance = self.video_cfg.camera_settings.distance
        self.camera.elevation = self.video_cfg.camera_settings.elevation
        self.camera.azimuth = self.video_cfg.camera_settings.azimuth

    def render_frame(self, data: mujoco.MjData):
        if self.video_cfg.enabled and self.renderer and self.camera:
            self.renderer.update_scene(data, camera=self.camera)
            self.frames.append(self.renderer.render())

    def save_video(self, fps: int):
        if not self.video_cfg.enabled:
            logger.info("Video recording disabled.")
            return

        if self.frames and self.renderer:
            output_video_path = Path(self.video_cfg.path)
            output_video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(output_video_path, self.frames, fps=fps)
            logger.info(
                f"Video saved: {output_video_path} ({len(self.frames)} frames, {fps} FPS)")
        elif self.renderer:  # Renderer exists but no frames
            logger.warning(
                "Video recording enabled, but no frames were recorded.")
        # No else needed if renderer was never initialized

    def close(self):
        if self.renderer:
            self.renderer.close()
