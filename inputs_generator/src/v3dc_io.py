from os import PathLike
from typing import Literal, Optional, TypedDict

import numpy as np

try:
    import video3d
except ImportError:
    print("The video3d library is not installed, check the tutorial on how to install it.")
    exit(1)

class View(TypedDict):
    """
    Object holding all information for a frame in a v3dc scan

    You can use it like any other dictionary, but IDEs like VSCode will use it
    to give hints about keys and datatypes.
    """

    idx:int
    """Frame index in the video"""

    name:str
    """Unique name for this frame, format 'frame%06d'"""

    img:Optional[np.ndarray]
    """Shape H,W,3, dtype UINT8, channels in RGB order"""

    depth:Optional[np.ndarray]
    """Shape H,W, dtype UINT16, 0 when invalid"""

    height:Optional[int]
    """img height"""

    width:Optional[int]
    """img width"""

    K:np.ndarray
    """Intrinsics, shape 3,3, dtype FLOAT32"""

    fx:float
    """Focal length in width"""

    fy:float
    """Focal length in height"""

    cx:float
    """Principle point x coordinate"""

    cy:float
    """Principle point y coordinate"""

    viewmat:np.ndarray
    """world-to-camera matrix, shape 4,4, dtype FLOAT32"""

    orientation:Literal['portrait','landscape','landscapeRight']
    """Image orientation"""

def read_v3dc_subset(
    fp:PathLike[str],
    frame_idxs:list[int],
    read_rgb:bool=True,
    read_depth:bool=True,
    orient_img:bool=True,
) -> list[View]:
    # Set up reader
    video3d_reader = video3d.Video3DReader(
        str(fp),
        video3d.READ_ALL if (read_rgb or read_depth) else video3d.READ_INFO,
        nr_preload_video=0,
        nr_preload_frames=0
    )

    # Memory check
    num_frames = len(frame_idxs)
    if (read_rgb or read_depth) and num_frames > 1500:
        raise ValueError("Attempting to load too many frames")

    # arg_sort sorts frame_idxs, arg_sort_inv is its inverse
    arg_sort = sorted(range(num_frames), key=lambda i: frame_idxs[i])
    arg_sort_inv = sorted(range(num_frames), key=lambda i: arg_sort[i])

    # Make sure frame_idxs is sorted (things break otherwise!)
    frame_idxs_sorted = [frame_idxs[i] for i in arg_sort]
    video3d_reader.seek_batch(frame_idxs_sorted)

    views = []
    for frame_idx in frame_idxs_sorted:
        frame = video3d_reader.getFrame()

        view = v3dc_frame_to_view(
            frame=frame,
            frame_idx=frame_idx,
            read_rgb=read_rgb,
            read_depth=read_depth,
            orient_img=orient_img,
        )

        views.append(view)

    # Make sure to convert back to the same ordering as `frame_idxs`
    views = [views[i] for i in arg_sort_inv]

    return views

def read_v3dc_sliced(
    fp:PathLike[str],
    start:int=0,
    end:Optional[int]=None,
    step:int=10,
    read_rgb:bool=True,
    read_depth:bool=True,
    orient_img:bool=True,
) -> list[View]:
    # Set up reader
    video3d_reader = video3d.Video3DReader(
        str(fp),
        video3d.READ_ALL if (read_rgb or read_depth) else video3d.READ_INFO
    )

    # Set step >= 10 to prevent loading too many images at a time
    if (read_rgb or read_depth) and step < 1 and ( end is None or end - start > 1500 ):
        raise ValueError("When reading RGB and/or Depth, make sure to limit the number of frames.")

    views = []
    for frame_idx,frame in enumerate(video3d_reader):
        if frame_idx < start or (frame_idx - start) % step != 0: continue
        if end is not None and frame_idx >= end: break

        view = v3dc_frame_to_view(
            frame=frame,
            frame_idx=frame_idx,
            read_rgb=read_rgb,
            read_depth=read_depth,
            orient_img=orient_img,
        )

        views.append(view)

    return views

def read_v3dc_iter(
    fp:PathLike[str],
    read_rgb:bool=True,
    read_depth:bool=True,
    orient_img:bool=True,
):
    # Set up reader
    video3d_reader = video3d.Video3DReader(
        str(fp),
        video3d.READ_ALL if (read_rgb or read_depth) else video3d.READ_INFO
    )

    for frame_idx,frame in enumerate(video3d_reader):
        view = v3dc_frame_to_view(
            frame=frame,
            frame_idx=frame_idx,
            read_rgb=read_rgb,
            read_depth=read_depth,
            orient_img=orient_img,
        )

        yield view


def v3dc_frame_to_view(
    frame,
    frame_idx:int,
    read_rgb:bool,
    read_depth:bool,
    orient_img:bool,
) -> View:
    info = frame.info()

    name = f'frame{frame_idx:06d}'

    # Get parameters
    intrinsics = np.squeeze(np.array(info['cameraIntrinsics'])).T
    c2w = np.squeeze(np.array(info['localToWorld'])).T
    w2c = np.linalg.inv(c2w) # world-to-view


    if read_rgb:
        img = frame.img().astype(np.uint8)[...,::-1] # RGB!

        height, width = img.shape[:2]

        if orient_img:
            if info['orientation'] in {'landscape', 'landscapeRight'}:
                pass # already oriented correctly
            elif info['orientation'] == 'portrait':
                img = np.rot90(img, k=-1)
            else:
                print(f"!WARNING! dont know what to do with image orientation {info['orientation']}")

        # FROM libwescan/modules/preprocess/src/Video3D2Pcd.cpp line 241
        if img.shape[1] != 1280:
            intrinsics[:2,:] /= 2

    else:
        img, height, width = None, None, None

    if read_depth:
        depth = np.squeeze(frame.depth()).astype(np.uint16)
        confidence = np.squeeze(frame.confidence())

        # Only keep confident depth
        # depth[confidence < 0.05] = 0

        if orient_img:
            if info['orientation'] in {'landscape', 'landscapeRight'}:
                pass # already oriented correctly
            elif info['orientation'] == 'portrait':
                depth = np.rot90(depth, k=-1)
            else:
                print(f"!WARNING! dont know what to do with image orientation {info['orientation']}")

    else:
        depth = None


    return {
        "idx": frame_idx,                               # int
        "name": name,                                   # str
        "img": img,                                     # array, np.uint8, shape H,W,3
        "depth": depth,                                 # array, np.uint16, shape H,W
        "height": height,                               # int
        "width": width,                                 # int
        "K": intrinsics,                                # array, np.float64, shape 3,3
        "fx": intrinsics[0,0], "fy": intrinsics[1,1],   # float
        "cx": intrinsics[0,2], "cy": intrinsics[1,2],   # float
        "viewmat": w2c,                                 # array, np.float64, shape 4,4
        "orientation": info['orientation'],             # str, probably one of portrait, landscape, landscapeRight (see above)
    }
