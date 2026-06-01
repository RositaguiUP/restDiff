import json
from pathlib import Path

ROOT_DIR = Path('/shared/3du_data')
ROOT_DIR_DATA = ROOT_DIR / "data"
ROOT_DIR_META = ROOT_DIR / "meta"


def get_envs():
    """Get list of all available `env_id`s for this dataset"""
    env_ids: list[str] = []

    for path in ROOT_DIR_META.glob("*.json"):
        if path.suffix == ".json":
            env_ids.append(path.stem)

    return sorted(env_ids)


class Environment:
    def __init__(self, env_id: str):
        self.id = env_id

        if not self.path_meta.exists():
            raise ValueError(f"Cannot find environment '{self.id}'")

        self._meta = None
        self._floor_numbers = None

    @property
    def meta(self):
        """
        Dictionary containing all metadata of a project. Open the file at
        `Environment.path_meta` to see the available fields. All projects
        have the same structured metadata.
        """
        if self._meta is None:
            with self.path_meta.open("r") as f:
                self._meta = json.load(f)
        return self._meta

    @property
    def path_meta(self):
        """Path to the metadata file `{DATABASE_ROOT}/meta/{ENV_ID}.json`"""
        return ROOT_DIR_META / f"{self.id}.json"

    @property
    def path_data(self):
        """
        Path to the Environment's raw data. This path is something of the form
        `{DATABASE_ROOT}/data/{YEAR}/{MONTH}/{PREFIX}/{ENV_ID}`
        where `PREFIX` is the first 2 characters of `ENV_ID`.

        See also the metadata `root_dir` field.
        """
        root_dir: str = self.meta[
            "root_dir"
        ]  # /data/uploaded_scans/{YEAR}/{MONTH}/{PREFIX}/{ENV_ID}
        assert root_dir.startswith("/data/uploaded_scans/")
        dir_prefix = root_dir[21:]  # len('/data/uploaded_scans/') == 21
        return ROOT_DIR_DATA / dir_prefix

    def get_path(self, subpath: str = ""):
        """Get a path inside the Environment's raw data folder."""
        return self.path_data / subpath.lstrip("/")

    @property
    def path_annotation(self):
        """Path to the `annotation.json` file."""
        return self.get_path("process/result/annotation.json")

    @property
    def floor_numbers(self):
        """
        Sorted list of floor numbers for this Environment.

        Value is cached after calling
        """
        if self._floor_numbers is None:
            floor_numbers: list[int] = []
            for dir in self.get_path("process/").glob("floor_*"):
                floor_numbers.append(int(dir.stem[6:]))  # len('floor_') == 6
            self._floor_numbers = sorted(floor_numbers)

        return self._floor_numbers

    def path_pcd(self, floor_number: int):
        """Path to the `pcd2.ply` file for the given floor number"""
        return self.get_path(f"process/floor_{floor_number}/pcd2.ply")

    def path_mesh(self, floor_number: int):
        """
        Path to the folder containing the `model.obj` file for the given floor
        number.

        Note: `model.obj` only contains the vertices and faces of the mesh, if
        you want a textured model, you need all other files in this directory.
        """
        return self.get_path(f"process/result/{floor_number}/3dmodel")

    def path_v3dc(self, floor_number: int):
        """Path to the `video.refine.v3dc` file for the given floor number"""
        return self.get_path(f"process/floor_{floor_number}/video.refine.v3dc")
    
    def path_sub_pano(self,floor_number: int):
        return self.get_path(f"process/result/{floor_number}/sub_models_pano/0.txt")

