import numpy as np
import fire
from pathlib import Path
import joblib


from loguru import logger


def pad_motion(motion_path: str, pad_length: int = 20):

    motion_path = Path(motion_path)
    data = joblib.load(motion_path)

    for name, motion in data.items():
        for k, v in motion.items():
            if not isinstance(v, np.ndarray):
                continue

            # pad along the first dimension with the last frame
            data[name][k] = np.concatenate(
                [v, v[-1:].repeat(pad_length, axis=0)], axis=0)

    new_motion_path = motion_path.parent / \
        f"{motion_path.stem}_padded_{pad_length}{motion_path.suffix}"
    joblib.dump(data, new_motion_path)
    logger.info(f"Saved padded motion to {new_motion_path}")


if __name__ == "__main__":
    fire.Fire(pad_motion)
