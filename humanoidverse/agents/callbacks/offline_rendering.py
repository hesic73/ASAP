import numpy as np
from humanoidverse.agents.callbacks.base_callback import RL_EvalCallback
from humanoidverse.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from humanoidverse.agents.ppo.ppo import PPO
from typing import List, Dict, Any, Optional
from pathlib import Path


class OfflineRenderingCallback(RL_EvalCallback):
    """
    Callback for offline rendering of evaluation episodes in IsaacGym and IsaacSim.
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
            training_loop (PPO): The PPO training loop instance.
        """
        super().__init__(config, training_loop)

        self.env: LeggedRobotBase = training_loop.env
        self.simulator_name = self.env.config.simulator.config.name

        assert self.simulator_name in ["isaacsim", "isaacgym"]

        # General rendering properties
        self.video_width = self.config.get("video_width", 1080)
        self.video_height = self.config.get("video_height", 1920)
        self.video_filename = Path(self.config.video_filename)
        self.video_filename.parent.mkdir(parents=True, exist_ok=True)

        # IsaacGym specific
        self.frames: List[np.ndarray] = []
        self.camera: Optional[Any] = None
        self.camera_offset_gym = self.config.get("camera_offset", [0.8, -0.8, 0.3])
        self.camera_rotation_axis_gym = self.config.get("camera_rotation_axis", [0.0, 0.0, 1])
        self.camera_rotation_angle_deg_gym = self.config.get("camera_rotation_angle_deg", 135)

        # IsaacSim specific
        self.camera_offset_sim = np.array(self.config.get("camera_offset", [0.8, -0.8, 0.3]))
        self.camera_prim_path_sim = self.config.get("camera_prim_path", "/OmniverseKit_Persp")
        self.video_writer_sim = None
        self.rgb_annotator_sim = None
        self.render_product_sim = None
        self.robot_articulation_sim = None


    def _setup_camera_isaacgym(self) -> None:
        """
        Sets up the camera sensor and attaches it to the robot's body for IsaacGym.
        """
        from isaacgym import gymapi
        gym = self.env.simulator.gym
        env_handle = self.env.simulator.envs[0]
        # sim = self.env.simulator.sim # Not used but available if needed

        camera_properties = gymapi.CameraProperties()
        camera_properties.width = self.video_width
        camera_properties.height = self.video_height
        self.camera = gym.create_camera_sensor(env_handle, camera_properties)

        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(*self.camera_rotation_axis_gym), np.deg2rad(
                self.camera_rotation_angle_deg_gym)
        )

        actor_handle = gym.get_actor_handle(env_handle, 0)
        body_handle = gym.get_actor_rigid_body_handle(
            env_handle, actor_handle, 0)

        gym.attach_camera_to_body(
            self.camera,
            env_handle,
            body_handle,
            gymapi.Transform(gymapi.Vec3(*self.camera_offset_gym), camera_rotation),
            gymapi.FOLLOW_POSITION,
        )

    def _capture_frame_isaacgym(self) -> None:
        """
        Captures a single frame from the camera and appends it to the frames list for IsaacGym.
        """
        if self.camera is None:
            raise RuntimeError(
                "Camera not initialized. Call _setup_camera_isaacgym() first.")

        from isaacgym import gymapi
        gym = self.env.simulator.gym
        sim = self.env.simulator.sim
        env_handle = self.env.simulator.envs[0]

        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)
        img = gym.get_camera_image(
            sim, env_handle, self.camera, gymapi.IMAGE_COLOR)
        img = np.reshape(img, (self.video_height, self.video_width, 4))
        self.frames.append(img[..., :3])  # Keep only RGB channels

    def _save_video_isaacgym(self) -> None:
        """
        Saves the captured frames as a video file for IsaacGym.
        """
        if not self.frames:
            print("No frames captured to save video.")
            return

        try:
            from moviepy.editor import ImageSequenceClip
        except ImportError:
            print(
                "moviepy not installed. Please install it to save videos: pip install moviepy")
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

    def _setup_camera_and_writer_isaacsim(self) -> None:
        """Initializes the replicator annotator and the video writer for IsaacSim."""
        import omni.replicator.core as rep
        import imageio
        from isaaclab.sim import SimulationContext # For type hinting

        self.sim: SimulationContext = self.env.simulator.sim
        self.robot_articulation_sim = self.env.simulator.scene.articulations["robot"]

        self.render_product_sim = rep.create.render_product(
            self.camera_prim_path_sim, (self.video_width, self.video_height)
        )
        self.rgb_annotator_sim = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self.rgb_annotator_sim.attach([self.render_product_sim])

        fps = int(1.0 / self.env.dt)
        self.video_writer_sim = imageio.get_writer(self.video_filename, fps=fps)

    def _update_camera_and_capture_frame_isaacsim(self) -> None:
        """Updates camera position to follow the robot and captures a single frame for IsaacSim."""
        base_pos = self.robot_articulation_sim.data.root_pos_w[0].cpu().numpy()
        eye_pos = base_pos + self.camera_offset_sim
        self.sim.set_camera_view(eye=eye_pos.tolist(), target=base_pos.tolist())

        # Step the replicator pipeline to capture the rendered data
        self.sim.render()
        rgb_data = self.rgb_annotator_sim.get_data()

        frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(self.video_height, self.video_width, 4)
        self.video_writer_sim.append_data(frame[..., :3])

    def _close_writer_and_cleanup_isaacsim(self) -> None:
        """Closes the video writer and cleans up replicator resources for IsaacSim."""
        self.video_writer_sim.close()
        # Detach annotator only if it was attached
        if self.render_product_sim and self.rgb_annotator_sim:
            self.rgb_annotator_sim.detach(self.render_product_sim)


    def on_pre_evaluate_policy(self):
        """
        Called before policy evaluation begins. Sets up the camera based on the simulator.
        """
        if self.simulator_name == "isaacgym":
            self.frames = []  # Clear frames from previous evaluations
            self._setup_camera_isaacgym()
        elif self.simulator_name == "isaacsim":
            self._setup_camera_and_writer_isaacsim()

    def on_post_eval_env_step(self, actor_state: Any):
        """
        Called after each environment step during evaluation. Captures a frame.

        Args:
            actor_state (Any): The state of the actor after the step (passed through).
        """
        if self.simulator_name == "isaacgym":
            self._capture_frame_isaacgym()
        elif self.simulator_name == "isaacsim":
            self._update_camera_and_capture_frame_isaacsim()
        return actor_state

    def on_post_evaluate_policy(self):
        """
        Called after policy evaluation ends. Saves the captured frames as a video.
        """
        if self.simulator_name == "isaacgym":
            self._save_video_isaacgym()
        elif self.simulator_name == "isaacsim":
            self._close_writer_and_cleanup_isaacsim()