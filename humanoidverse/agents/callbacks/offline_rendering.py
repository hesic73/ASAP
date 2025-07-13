import numpy as np
from isaacgym import gymapi
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO
from typing import List, Dict, Any, Optional

class OfflineRenderingCallback(RL_EvalCallback):
    """
    Callback for offline rendering of evaluation episodes in IsaacGym.
    Captures camera frames and saves them as a video.
    """

    def __init__(self, config: Dict[str, Any], training_loop: PPO):
        """
        Initializes the OfflineRenderingCallback.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the callback.
                                     Expected keys:
                                     - video_width (int): Width of the output video.
                                     - video_height (int): Height of the output video.
                                     - camera_offset (List[float]): XYZ offset of the camera from the followed body.
                                     - camera_rotation_axis (List[float]): Axis for camera rotation (e.g., [-0.3, 0.2, 1]).
                                     - camera_rotation_angle_deg (float): Angle in degrees for camera rotation.
                                     - camera_follow_mode (str): Camera follow mode ("FOLLOW_POSITION", "FOLLOW_TRANSFORM").
            training_loop (PPO): The PPO training loop instance.
        """
        super().__init__(config, training_loop)
        self.env: LeggedRobotBase = training_loop.env
        self.frames: List[np.ndarray] = []
        self.camera: Optional[gymapi.CameraSensor] = None

        # Set default camera properties if not provided in config
        self.video_width = self.config.get("video_width", 1080)
        self.video_height = self.config.get("video_height", 1920)
        self.camera_offset = gymapi.Vec3(*self.config.get("camera_offset", [0.8, -0.8, 0.3]))
        self.camera_rotation_axis = gymapi.Vec3(*self.config.get("camera_rotation_axis", [0.0, 0.0, 1]))
        self.camera_rotation_angle_deg = self.config.get("camera_rotation_angle_deg", 135)
        self.camera_follow_mode_str = self.config.get("camera_follow_mode", "FOLLOW_POSITION")
        self.video_filename = self.config.get("video_filename", "eval.mp4")

        # Map string to gymapi constant
        self.camera_follow_mode = self._get_camera_follow_mode(self.camera_follow_mode_str)

    def _get_camera_follow_mode(self, mode_str: str) -> int:
        """Converts a string representation of camera follow mode to gymapi constant."""
        if mode_str == "FOLLOW_POSITION":
            return gymapi.FOLLOW_POSITION
        elif mode_str == "FOLLOW_TRANSFORM":
            return gymapi.FOLLOW_TRANSFORM
        else:
            raise ValueError(f"Unknown camera follow mode: {mode_str}. Expected 'FOLLOW_POSITION' or 'FOLLOW_TRANSFORM'.")

    def _setup_camera(self) -> None:
        """
        Sets up the camera sensor and attaches it to the robot's body.
        """
        gym = self.env.simulator.gym
        env_handle = self.env.simulator.envs[0]
        sim = self.env.simulator.sim

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.video_width
        camera_properties.height = self.video_height
        self.camera = gym.create_camera_sensor(env_handle, camera_properties)

        camera_rotation = gymapi.Quat.from_axis_angle(
            self.camera_rotation_axis, np.deg2rad(self.camera_rotation_angle_deg)
        )

        actor_handle = gym.get_actor_handle(env_handle, 0)
        body_handle = gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0)

        gym.attach_camera_to_body(
            self.camera,
            env_handle,
            body_handle,
            gymapi.Transform(self.camera_offset, camera_rotation),
            self.camera_follow_mode,
        )

    def _capture_frame(self) -> None:
        """
        Captures a single frame from the camera and appends it to the frames list.
        """
        if self.camera is None:
            raise RuntimeError("Camera not initialized. Call _setup_camera() first.")

        gym = self.env.simulator.gym
        sim = self.env.simulator.sim
        env_handle = self.env.simulator.envs[0]

        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, env_handle, self.camera, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (self.video_height, self.video_width, 4))
        self.frames.append(img[..., :3]) # Keep only RGB channels

    def _save_video(self) -> None:
        """
        Saves the captured frames as a video file.
        """
        if not self.frames:
            print("No frames captured to save video.")
            return

        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print("moviepy not installed. Please install it to save videos: pip install moviepy")
            return

        fps = int(1.0 / self.env.dt)
        print(f"Saving video with FPS: {fps}")
        clip = ImageSequenceClip(self.frames, fps=fps)
        clip.write_videofile(
            self.video_filename,
            codec="libx264",
            audio=False,
            threads=4,
        )
        print(f"Video saved to {self.video_filename}")

    def on_pre_evaluate_policy(self):
        """
        Called before policy evaluation begins. Sets up the camera.
        """
        self.frames = []  # Clear frames from previous evaluations
        self._setup_camera()

    def on_post_eval_env_step(self, actor_state: Any):
        """
        Called after each environment step during evaluation. Captures a frame.

        Args:
            actor_state (Any): The state of the actor after the step (passed through).
        """
        self._capture_frame()
        return actor_state

    def on_post_evaluate_policy(self):
        """
        Called after policy evaluation ends. Saves the captured frames as a video.
        """
        self._save_video()