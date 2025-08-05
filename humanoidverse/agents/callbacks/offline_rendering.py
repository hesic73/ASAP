import numpy as np
import imageio
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO
from typing import List, Dict, Any, Optional
from pathlib import Path

from loguru import logger


class OfflineRenderingCallback(RL_EvalCallback):
    """
    Callback for offline rendering of evaluation episodes in IsaacGym, IsaacSim, and Genesis.
    Captures camera frames and saves them as a video using a unified interface.
    """

    def __init__(self, config: Dict[str, Any], training_loop: PPO):
        """
        Initializes the OfflineRenderingCallback.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the callback.
                                     Expected keys:
                                     - video_width (int): Width of the output video.
                                     - video_height (int): Height of the output video.
                                     - camera_offset (List[float]): XYZ offset of the camera.
                                     - video_filename (str): Path to save the video.
            training_loop (PPO): The PPO training loop instance.
        """
        super().__init__(config, training_loop)

        self.env: LeggedRobotBase = training_loop.env
        self.simulator_name = self.env.config.simulator.config.name
        assert self.simulator_name in ["isaacsim", "isaacgym", "genesis"]

        # General rendering properties
        self.video_width = self.config.get("video_width", 1080)
        self.video_height = self.config.get("video_height", 1920)
        self.video_filename = Path(self.config.video_filename)
        self.video_filename.parent.mkdir(parents=True, exist_ok=True)
        self.fps = int(1.0 / self.env.dt)
        self.camera_offset = np.array(self.config.get("camera_offset", [0.8, -0.8, 0.3]))
        self.frames: List[np.ndarray] = []

        # IsaacGym specific
        self.camera_gym: Optional[Any] = None

        # IsaacSim specific
        self.camera_prim_path_sim = self.config.get("camera_prim_path", "/OmniverseKit_Persp")
        self.rgb_annotator_sim = None
        self.render_product_sim = None
        self.robot_articulation_sim = None

        # Genesis specific
        self.camera_gen: Optional[Any] = None

    def _setup_camera_isaacgym(self) -> None:
        """Sets up the camera sensor for IsaacGym."""
        from isaacgym import gymapi
        gym = self.env.simulator.gym
        env_handle = self.env.simulator.envs[0]

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.video_width
        camera_properties.height = self.video_height
        self.camera_gym = gym.create_camera_sensor(env_handle, camera_properties)

        camera_transform = gymapi.Transform()
        camera_transform.p = gymapi.Vec3(*self.camera_offset)
        camera_transform.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0.0, 0.0, 1.0), np.arctan2(self.camera_offset[0], self.camera_offset[1])
        )

        actor_handle = gym.get_actor_handle(env_handle, 0)
        body_handle = gym.get_actor_rigid_body_handle(env_handle, actor_handle, 0)

        gym.attach_camera_to_body(
            self.camera_gym, env_handle, body_handle, camera_transform, gymapi.FOLLOW_POSITION
        )

    def _capture_frame_isaacgym(self) -> None:
        """Captures a single frame from the camera in IsaacGym."""
        from isaacgym import gymapi
        gym = self.env.simulator.gym
        sim = self.env.simulator.sim
        env_handle = self.env.simulator.envs[0]

        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(sim, env_handle, self.camera_gym, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (self.video_height, self.video_width, 4))
        self.frames.append(img[..., :3])  # Keep only RGB channels

    def _setup_camera_isaacsim(self) -> None:
        """Initializes the replicator annotator for IsaacSim."""
        import omni.replicator.core as rep
        from isaaclab.sim import SimulationContext  # For type hinting

        self.sim: SimulationContext = self.env.simulator.sim
        self.robot_articulation_sim = self.env.simulator.scene.articulations["robot"]

        self.render_product_sim = rep.create.render_product(
            self.camera_prim_path_sim, (self.video_width, self.video_height)
        )
        self.rgb_annotator_sim = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb_annotator_sim.attach([self.render_product_sim])

    def _update_camera_and_capture_frame_isaacsim(self) -> None:
        """Updates camera position and captures a single frame for IsaacSim."""
        base_pos = self.robot_articulation_sim.data.root_pos_w[0].cpu().numpy()
        eye_pos = base_pos + self.camera_offset
        self.sim.set_camera_view(eye=eye_pos.tolist(), target=base_pos.tolist())

        self.sim.render()
        rgb_data = self.rgb_annotator_sim.get_data()
        frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self.video_height, self.video_width, 4)
        self.frames.append(frame[..., :3])

    def _setup_camera_genesis(self) -> None:
        """Gets the pre-configured camera from the Genesis scene."""
        cameras = self.env.simulator.scene.visualizer.cameras
        assert len(cameras) == 1
        self.camera_gen = cameras[0]

    def _update_camera_and_capture_frame_genesis(self) -> None:
        """Updates camera position and captures a single frame for Genesis."""
        robot = self.env.simulator.robot
        robot_pos = robot.get_pos().squeeze(0).cpu().numpy()
        eye_pos = robot_pos + self.camera_offset
        self.camera_gen.set_pose(pos=eye_pos, lookat=robot_pos)
        rgb_arr, _, _, _ = self.camera_gen.render()
        self.frames.append(rgb_arr)

    def _save_video(self) -> None:
        """Saves the captured frames as a video file using imageio."""
        logger.info(f"Saving video with {len(self.frames)} frames at {self.fps} FPS.")
        with imageio.get_writer(self.video_filename, fps=self.fps) as writer:
            for frame in self.frames:
                writer.append_data(frame)
        logger.info(f"Video saved to {self.video_filename}")

    def on_pre_evaluate_policy(self):
        """Called before policy evaluation begins. Sets up the camera."""
        self.frames = []  # Clear frames from previous evaluations
        if self.simulator_name == "isaacgym":
            self._setup_camera_isaacgym()
        elif self.simulator_name == "isaacsim":
            self._setup_camera_isaacsim()
        elif self.simulator_name == "genesis":
            self._setup_camera_genesis()

    def on_post_eval_env_step(self, actor_state: Any):
        """Called after each environment step during evaluation to capture a frame."""
        if self.simulator_name == "isaacgym":
            self._capture_frame_isaacgym()
        elif self.simulator_name == "isaacsim":
            self._update_camera_and_capture_frame_isaacsim()
        elif self.simulator_name == "genesis":
            self._update_camera_and_capture_frame_genesis()
        return actor_state

    def on_post_evaluate_policy(self):
        """Called after policy evaluation ends to save the video."""
        if not self.frames:
            logger.warning("No frames were captured to save video.")
            return

        self._save_video()

        # Specific cleanup for isaacsim
        if self.simulator_name == "isaacsim":
            if self.render_product_sim and self.rgb_annotator_sim:
                self.rgb_annotator_sim.detach(self.render_product_sim)