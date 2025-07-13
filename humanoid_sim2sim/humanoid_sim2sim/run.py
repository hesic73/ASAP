import hydra
from omegaconf import DictConfig, OmegaConf


from humanoid_sim2sim.utils.observation_manager import ObservationManager
import numpy as np

from humanoid_sim2sim.consts import CONFIG_DIR


@hydra.main(version_base="1.1", config_path=CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    obs_manager = ObservationManager(cfg.obs)
    obs_manager.update(
        {
            'actions': np.random.rand(23).astype(np.float32),
            'base_ang_vel': np.random.rand(3).astype(np.float32),
            "dif_local_rigid_body_pos": np.random.rand(3 * 24).astype(np.float32),
            "local_ref_rigid_body_pos": np.random.rand(3 * 24).astype(np.float32),
            "dof_pos": np.random.rand(23).astype(np.float32),
            "dof_vel": np.random.rand(23).astype(np.float32),
            "projected_gravity": np.random.rand(3).astype(np.float32),
        }
    )

    initial_obs = obs_manager.get()
    print("Initial observation vector:", initial_obs.shape)


if __name__ == "__main__":
    main()
