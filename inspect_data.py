from sympy import root
import tyro
import joblib

from loguru import logger

def main(filename: str):
    """
    Inspect the data in a file.
    """
    data=joblib.load(filename)
    name = list(data.keys())
    assert len(name) == 1
    name = name[0]
    data = data[name]

    logger.info(f"Data loaded from {filename}, name: {name}")

    pose_aa=data['pose_aa']
    root_trans_offset=data['root_trans_offset']
    print(pose_aa.shape)
    print(root_trans_offset.shape)
    print(pose_aa[0])

if __name__ == "__main__":
    tyro.cli(main)