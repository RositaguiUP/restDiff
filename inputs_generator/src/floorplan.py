from dataclasses import dataclass
from typing import Literal
from plyfile import PlyData,PlyElement
import json

import numpy as np
import os 


def normalize(a:np.ndarray) -> np.ndarray:
    """Normalize vector"""
    return a / np.linalg.norm(a)


def transform(vert, P):
    vert = np.hstack((vert,np.ones((vert.shape[0],1)))) @ P

    return vert[:,:3].copy() / vert[:,3:4]


# Default vertex indices for a box
FACES = np.array([
    [0, 2, 1],
    [2, 3, 1],
    [1, 3, 5],
    [3, 7, 5],
    [4, 5, 7],
    [4, 7, 6],
    [0, 4, 6],
    [6, 2, 0],
    [3, 2, 6],
    [6, 7, 3],
    [1, 5, 4],
    [1, 4, 0],
])

@dataclass
class Opening:
    """
    Opening dataclass, containing all info about an opening in a wall
    """
    type:str
    width:float
    height:float
    z:float
    t:float

    def vertices(self, parent:'Wall'):
        # Use *1.1 to prevent Z-fighting
        d0 = -1.1*parent.thickness * parent.balance
        d1 = 1.1*parent.thickness * (1-parent.balance)
        s = self.t * parent.width - self.width/2
        return transform(np.array([
            [s,            self.z,             d0 ],
            [s,            self.z,             d1 ],
            [s,            self.z+self.height, d0 ],
            [s,            self.z+self.height, d1 ],
            [s+self.width, self.z,             d0 ],
            [s+self.width, self.z,             d1 ],
            [s+self.width, self.z+self.height, d0 ],
            [s+self.width, self.z+self.height, d1 ],
        ]), parent.wall_to_world)

@dataclass
class Wall:
    """
    Wall dataset, containing all info about a single wall
    """
    openings:list[Opening]
    ax:float
    ay:float
    az:float
    aheight:float
    bx:float
    by:float
    bz:float
    bheight:float
    thickness:float
    balance:float
    identifier:str

    def __post_init__(self):
        self.a = np.array([self.ax, 0, self.ay])
        self.b = np.array([self.bx, 0, self.by])

        # Wall-to-World transformation matrix
        # Moves coordinats on the wall to world coordinates
        P:np.ndarray = np.eye(4)
        P[:3,0] = normalize(self.b - self.a) # X-axis: from point A to B
        P[:3,1] = np.array([0,1,0]) # Y-axis: Up
        P[:3,2] = normalize( np.cross(P[:3,1], P[:3,0]) ) # Z-axis: remaining axis
        P[3,:3] = self.a # Origin: point A
        self.wall_to_world = P

        # World-to-Wall transformation matrix
        self.world_to_wall = np.linalg.inv(P)

        # Length of the wall measured parallel to the floor
        self.width:float = np.linalg.norm(self.b - self.a) # type: ignore

        d0 = -self.thickness * self.balance
        d1 = self.thickness * (1-self.balance)
        self.vertices = transform(np.array([
            [0,          self.az,              d0 ],
            [0,          self.az,              d1 ],
            [0,          self.az+self.aheight, d0 ],
            [0,          self.az+self.aheight, d1 ],
            [self.width, self.bz,              d0 ],
            [self.width, self.bz,              d1 ],
            [self.width, self.bz+self.bheight, d0 ],
            [self.width, self.bz+self.bheight, d1 ],
        ]), self.wall_to_world)

        self.min_z = min(self.az, self.bz)
        self.max_z = max(self.az+self.aheight, self.bz+self.bheight)
        self.height = self.max_z - self.min_z

    def to_mesh(self) -> tuple[np.ndarray,np.ndarray]:
        """
        Convert a wall with openings to a mesh.
        Wall vertices have a red color, opening vertices are marked in green.
        """
        vertices = []
        faces = []

        # Add the wall vertices/faces
        vert = self.vertices
        color = np.array([[215,215,215]]) #np.random.randint(0,255,(1,3))
        vert = np.hstack([vert, np.repeat(color, vert.shape[0], 0)])
        vertices.append(vert)
        faces.append(FACES.copy())

        # Keep track of the face index offset for future vertices
        idxs_offset = vert.shape[0]

        # Add the openings to the set of vertices and edges
        for opening in self.openings:

            # Get colored vertices
            vert = opening.vertices(self)
            vert = np.hstack([vert, np.repeat([[50,50,50]], vert.shape[0], 0)])
            vertices.append(vert)

            # Add face indices, offset to account for previous vertices
            faces.append(FACES.copy() + idxs_offset)
            idxs_offset += vert.shape[0]

        # Concatenate vertices and faces together
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        return vertices, faces

@dataclass
class FloorPlan:
    """
    Annotation of a floor
    """
    walls:list[Wall]
    mesh_to_scan:np.ndarray
    floorplan_to_mesh:np.ndarray
    name:str
    number:int


    @classmethod
    def read(cls, path_json:str, floor_number:int,sub_pano_path:str):
        with open(path_json, 'r') as f:
            obj = json.load(f)

        for floor_obj in obj.get('floors',[]):
            # Floors without a scan (ie manually annotated) have no floorNumber property
            if floor_obj.get('floorNumber',None) == floor_number:
                return cls.from_obj(floor_obj,sub_pano_path)

        raise ValueError(f"Floor number {floor_number} not found in annotation file {path_json}")


    @classmethod
    def from_obj(cls, obj:dict,sub_pano_path:str):
        """
        Args
        ---
        - :param obj dict: The object from the annotation json file for this floor
        """

        name = obj['floorName']
        number = obj['floorNumber']

        # Read all the walls from the json object
        walls = []
        for i,wall in enumerate(obj['floorplan']['walls']):

            # First add each opening
            openings = []
            for opening in wall['openings']:
                # Openings have the same fields as the json file, so nothing fancy needs to happen here
                openings.append(Opening(
                    type=opening['type'],
                    width=opening['width'] / 100.,
                    height=opening['z_height'] / 100.,
                    z=opening['z'] / 100.,
                    t=opening['t'],
                ))

            # Wall has about the same fields as the json file, so nothing fancy needs to happen here
            walls.append(Wall(
                openings=openings,
                ax=wall['a']['x'] / 100.,
                ay=wall['a']['y'] / 100.,
                az=wall['az']['z'] / 100.,
                aheight=wall['az']['h'] / 100.,
                bx=wall['b']['x'] / 100.,
                by=wall['b']['y'] / 100.,
                bz=wall['bz']['z'] / 100.,
                bheight=wall['bz']['h'] / 100.,
                thickness=wall['thickness'] / 100.,
                balance=wall['balance'],
                identifier=f"wall_{i}"
            ))

        if 'manhattan_pose' in obj:
            manhattan_pose = np.array(obj['manhattan_pose']).reshape(4,4)
        else:
            print("!WARNING! 'manhattan_pose' not found in obj, assuming Identity transformation")
            manhattan_pose = np.eye(4)
        
        if os.path.exists(sub_pano_path):
            sub_pano_matrix = np.loadtxt(sub_pano_path)
            sub_pano_matrix = sub_pano_matrix.reshape(4,4)
        else:
            print("!WARNING! sub_pano path not found ")
            sub_pano_matrix = np.eye(4)

        pose = np.array(obj['pose']).reshape(4,4)
        floorBottom = obj.get('_floorBottom', 0.) / 100. # covert to cm!

        # Scan-to-mesh transformation: manhattan pose, but with X and Z axis flipped
        # scan_to_mesh = manhattan_pose @ np.diag([-1, 1, -1, 1])
        scan_to_mesh = sub_pano_matrix
        
        mesh_to_scan = np.linalg.inv(scan_to_mesh)

        # pose is mesh-to-floorplan transformation, but does not yet include the floorBottom!
        # floorBottom overwrites the Y-translation of the floorplan-to-mesh transformation
        # So, first compute floorplan-to-mesh
        floorplan_to_mesh = np.linalg.inv( pose )
        # Overwrite the Y coordinate of the translation vector with floorBottom
        floorplan_to_mesh[1,3] = floorBottom

        return cls(name=name, number=number, walls=walls, floorplan_to_mesh=floorplan_to_mesh.T, mesh_to_scan=mesh_to_scan.T)


    def to_mesh(self, system:Literal['self','mesh','pcd']='mesh') -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a floor to a collection of vertices and faces

        Parameter 'system' allows a transformation to align with either mesh or pointcloud
        """
        vertices = []
        faces = []

        # Make sure to offset the index of each face when taking the union of vertices!
        idxs_offset = 0

        for wall in self.walls:
            # Vertices and faces of the wall
            vert,fac = wall.to_mesh()

            # Vertices can just be added to the total set
            vertices.append(vert)

            # Make sure to offset the face indices
            faces.append(fac + idxs_offset)

            # Save offset for the next wall
            idxs_offset += vert.shape[0]

        # Concatenate vertices and faces
        vertices = np.concatenate(vertices)

        # Move vertices to pointcloud coordinates
        if system == 'mesh':
            vertices[:,:3] = transform(vertices[:,:3], self.floorplan_to_mesh)

        elif system == 'pcd':
            vertices[:,:3] = transform(vertices[:,:3], self.floorplan_to_mesh @ self.mesh_to_scan)

        elif system != 'self':
            raise ValueError(f"Unkown system {system}")

        faces = np.concatenate(faces)
        return vertices, faces


    def save_mesh(self, save_to:str, system:Literal['self','mesh','pcd']='mesh'):

        # Iterate the specified floors...
        vertices,faces = self.to_mesh(system=system)

        # Convert to a format expected by PlyData
        vertex = np.array(
            [tuple(vert) for vert in vertices],
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        )
        face = np.array(
            [(face.tolist(),) for face in faces],
            dtype=[('vertex_indices', 'i4', (3,))]
        )

        PlyData(
            [
                PlyElement.describe(vertex, 'vertex'),
                PlyElement.describe(face, 'face')
            ],
            text=True
        ).write(save_to)

if __name__ == '__main__':

    # add the parent of this file's folder to the path. Likely /shared/3du_data
    # this allows `import src` to refer to `/shared/3du_data/src`
    import sys
    from pathlib import Path

    path_insert = str(Path(__file__).resolve().parent.parent)
    sys.path.insert(0, path_insert)
    print(f"Added '{path_insert}' to your $PATH")

    from utils.v3dc_io import read_v3dc_sliced, View
    from utils.environment import Environment

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", required=True, type=str, help="Project ID")
    parser.add_argument("--floor-number", type=int, help="(Optional) what floor number to use, defaults to the first")
    parser.add_argument("--path-out", required=True, type=Path, help="Path to save to")
    parser.add_argument("--system", default="mesh", choices=["self","mesh","pcd"], help="Coordinate system in which to export")
    args = parser.parse_args()

    print("Reading data ...")
    env = Environment(args.env_id)
    floor_number = args.floor_number if args.floor_number is not None else env.floor_numbers[0]

    sub_pano_path = env.path_sub_pano(floor_number)

    print("Getting floorplan ...")
    fp = FloorPlan.read(env.path_annotation, floor_number=floor_number,sub_pano_path=sub_pano_path)

    print("Saving floorplan ...")
    fp.save_mesh(args.path_out, system=args.system)
    print(f"Saved to {args.path_out}")

    if args.system == "mesh":
        print("The exported mesh should align with the project mesh at", env.path_mesh(floor_number=floor_number))
    elif args.system == "pcd":
        print("The exported mesh should align with the pointcloud", env.path_pcd(floor_number=floor_number))
