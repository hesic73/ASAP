import numpy as np
import fire
from pathlib import Path
import joblib

from loguru import logger


def truncate_motion(motion_path: str, truncate_length: int):
    motion_path = Path(motion_path)
    data = joblib.load(motion_path)

    for name, motion in data.items():
        for k, v in motion.items():
            if not isinstance(v, np.ndarray):
                continue

            # truncate along the first dimension
            data[name][k] = v[:truncate_length]

    new_motion_path = motion_path.parent / \
        f"{motion_path.stem}_truncated_{truncate_length}{motion_path.suffix}"
    joblib.dump(data, new_motion_path)
    logger.info(f"Saved truncated motion to {new_motion_path}")


if __name__ == "__main__":
    fire.Fire(truncate_motion)
