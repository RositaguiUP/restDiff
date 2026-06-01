# Default Configuration values and structure of Configuration Node

from yacs.config import CfgNode as CN

_C = CN()

# Project level configuration
_C.scene_name = "6VSV7_695"  # No spaces
_C.img_size = 512

# Output paths
_C.output_path = "data"

# Inputs generator
_C.env_id = ""
_C.floor_numbers = [0]
_C.dist_thresh = 0.15
_C.rot_thresh = 10.0
_C.blur_thresh = 15.0
_C.step = 1

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
