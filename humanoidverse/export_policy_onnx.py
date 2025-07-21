import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from humanoidverse.utils.logging import HydraLoggerBridge
import logging
from utils.config_utils import *  # noqa: E402, F403

from humanoidverse.utils.config_utils import *  # noqa: E402, F403
from loguru import logger


@hydra.main(config_path="config", config_name="base_eval", version_base="1.1")
def main(override_config: OmegaConf):
    OmegaConf.resolve(override_config)
    # logging to hydra log file
    hydra_log_path = os.path.join(
        HydraConfig.get().runtime.output_dir, "export.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")

    # Get log level from LOGURU_LEVEL environment variable or use INFO as default
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    os.chdir(hydra.utils.get_original_cwd())

    if override_config.checkpoint is not None:
        has_config = True
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
            if not config_path.exists():
                has_config = False
                logger.error(f"Could not find config path: {config_path}")

        if has_config:
            logger.info(f"Loading training config file from {config_path}")
            with open(config_path) as file:
                train_config = OmegaConf.load(file)

            if train_config.eval_overrides is not None:
                train_config = OmegaConf.merge(
                    train_config, train_config.eval_overrides
                )

            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.eval_overrides is not None:
            config = override_config.copy()
            eval_overrides = OmegaConf.to_container(
                config.eval_overrides, resolve=True)
            for arg in sys.argv[1:]:
                if not arg.startswith("+"):
                    key = arg.split("=")[0]
                    if key in eval_overrides:
                        del eval_overrides[key]
            config.eval_overrides = OmegaConf.create(eval_overrides)
            config = OmegaConf.merge(config, eval_overrides)
        else:
            config = override_config

    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import export_policy_as_onnx

    pre_process_config(config)

    # use config.device if specified, otherwise use cuda if available
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    export_log_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info(f"Saving export logs to {export_log_dir}")
    with open(export_log_dir / "config.yaml", "w") as file:
        OmegaConf.save(config, file)

    # Create a dummy environment class
    class DummyEnv:
        def __init__(self, config):
            self.config = config
            self.num_envs = config.num_envs

        def _calculate_history_dim(self, history_name):
            """Calculate the total dimension for a history observation"""
            if history_name not in self.config.obs.obs_auxiliary:
                return 0

            history_config = self.config.obs.obs_auxiliary[history_name]
            total_dim = 0
            for key, history_length in history_config.items():
                if key in self.config.obs.obs_dims:
                    total_dim += self.config.obs.obs_dims[key] * history_length
            return total_dim

        def reset_all(self):

            obs_dict = {}

            # Generate dummy observations for each obs_dict entry
            for obs_key, obs_config in self.config.obs.obs_dict.items():
                obs_tensors = []

                for key in sorted(obs_config):
                    if key.startswith('history_'):
                        # Handle history observations
                        dim = self._calculate_history_dim(key)
                    else:
                        # Handle regular observations
                        if key in self.config.obs.obs_dims:
                            dim = self.config.obs.obs_dims[key]
                        else:
                            # Skip if not found in obs_dims
                            continue

                    # Create dummy tensor with shape (num_envs, dim)
                    dummy_tensor = torch.zeros(
                        self.num_envs, dim, dtype=torch.float32)
                    obs_tensors.append(dummy_tensor)

                # Concatenate all tensors for this obs_key
                if obs_tensors:
                    obs_dict[obs_key] = torch.cat(obs_tensors, dim=-1)

            return obs_dict

    env = DummyEnv(config.env.config)

    # Initialize algorithm and load checkpoint
    algo: BaseAlgo = instantiate(
        config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    # Export policy to ONNX
    checkpoint_path = str(checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create export directory
    ROBOVERSE_ROOT_DIR = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    exported_policy_path = os.path.join(
        ROBOVERSE_ROOT_DIR, checkpoint_dir, 'exported')
    os.makedirs(exported_policy_path, exist_ok=True)

    exported_policy_name = checkpoint_path.split('/')[-1]
    exported_onnx_name = exported_policy_name.replace('.pt', '.onnx')

    # Get example observations and export to ONNX
    logger.info("Getting example observations for ONNX export...")
    example_obs_dict = algo.get_example_obs()

    logger.info("Exporting policy to ONNX format...")
    export_policy_as_onnx(
        algo.inference_model, exported_policy_path, exported_onnx_name, example_obs_dict)

    logger.info(
        f'Successfully exported policy as ONNX to: {os.path.join(exported_policy_path, exported_onnx_name)}')


if __name__ == "__main__":
    main()
