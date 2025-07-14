import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np

import onnxruntime

import mujoco

from loguru import logger

from humanoid_sim2sim.utils.observation_manager import ObservationManager
from humanoid_sim2sim.utils.camera_manager import CameraManager


from humanoid_sim2sim.consts import CONFIG_DIR, ASSETS_DIR


@hydra.main(version_base="1.1", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    model = mujoco.MjModel.from_xml_path(
        os.path.join(ASSETS_DIR, cfg.robot.asset.xml_path))
    data = mujoco.MjData(model)
    simulation_dt: float = cfg.simulation_dt
    model.opt.timestep = simulation_dt

    obs_manager = ObservationManager(cfg.obs)
    obs_manager.update(
        {
            'actions': np.random.rand(23).astype(np.float32),
            'base_ang_vel': np.random.rand(3).astype(np.float32),
            "dof_pos": np.random.rand(23).astype(np.float32),
            "dof_vel": np.random.rand(23).astype(np.float32),
            "projected_gravity": np.random.rand(3).astype(np.float32),
            "ref_motion_phase": np.random.rand(1).astype(np.float32),
        }
    )

    initial_obs = obs_manager.get()
    print("Initial observation vector:", initial_obs.shape)


if __name__ == "__main__":
    main()
