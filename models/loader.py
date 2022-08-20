import os
from matplotlib.pyplot import axis
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from typing_extensions import Self
import torch
from torch.utils.data import Dataset
import copy
import open3d as o3d

def fixedNumDownSample(vertices, desiredNumOfPoint, leftVoxelSize, rightVoxelSize):
    """ Use the method voxel_down_sample defined in open3d and do bisection iteratively 
        to get the appropriate voxel_size which yields the points with the desired number.
        INPUT:
            vertices: numpy array shape (n,3)
            desiredNumOfPoint: int, the desired number of points after down sampling
            leftVoxelSize: float, the initial bigger voxel size to do bisection
            rightVoxelSize: float, the initial smaller voxel size to do bisection
        OUTPUT:
            downSampledVertices: down sampled points with the original data type
    
    """
    assert leftVoxelSize > rightVoxelSize, "leftVoxelSize should be larger than rightVoxelSize"
    assert vertices.shape[0] > desiredNumOfPoint, "desiredNumOfPoint should be less than or equal to the num of points in the given array."
    if vertices.shape[0] == desiredNumOfPoint:
        return vertices
    
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(leftVoxelSize)
    assert len(pcd.points) <= desiredNumOfPoint, "Please specify a larger leftVoxelSize."
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd = pcd.voxel_down_sample(rightVoxelSize)
    assert len(pcd.points) >= desiredNumOfPoint, "Please specify a smaller rightVoxelSize."
    
    pcd.points = o3d.utility.Vector3dVector(vertices)
    midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
    pcd = pcd.voxel_down_sample(midVoxelSize)
    while len(pcd.points) != desiredNumOfPoint:
        if len(pcd.points) < desiredNumOfPoint:
            leftVoxelSize = copy.copy(midVoxelSize)
        else:
            rightVoxelSize = copy.copy(midVoxelSize)
        midVoxelSize = (leftVoxelSize + rightVoxelSize) / 2.
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd = pcd.voxel_down_sample(midVoxelSize)
    
    # print("final voxel size: ", midVoxelSize)
    downSampledVertices = np.asarray(pcd.points, dtype=vertices.dtype)
    return downSampledVertices

def quat_to_matrix9D(quat):
    """
    Euler angle to 3x3 rotation matrix.

    Args:
        euler (ndarray): Euler angle. Shape: (..., 3)
        order (str, optional):
            Euler rotation order AND order of parameter euler.
            E.g. "yxz" means parameter "euler" is (y, x, z) and
            rotation order is z applied first, then x, finally y.
            i.e. p' = YXZp, where p' and p are column vectors.
            Defaults to "zyx".
        unit (str, optional):
            Can be either degrees and radians.
            Defaults to degrees.
    """
    mat = np.identity(3)
    qw, qx, qy, qz = quat[...,0:1],quat[...,1:2],quat[...,2:3],quat[...,3:4]

    qxx = qx * qx
    qyy = qy * qy
    qzz = qz * qz

    qxy = qx * qy
    qxz = qx * qz
    qxw = qx * qw
    qyz = qy * qz
    qyw = qy * qw
    qzw = qz * qw

    mat=np.concatenate([
        1 - 2 * (qyy + qzz), 2 * (qxy - qzw), 2 * (qxz + qyw),
        2 * (qxy + qzw), 1 - 2 * (qxx + qzz), 2 * (qyz - qxw),
        2 * (qxz - qyw), 2 * (qyz + qxw), 1 - 2 * (qxx + qyy)
    ], axis=-1)

    mat = mat.reshape(*mat.shape[:-1], 3, 3)

    return mat



class BvhDataSet(Dataset):
    def __init__(self, bvh_folder,device="cuda:0", dtype=torch.float32,complete_flag=True,augment_flag=True):
        """
        Bvh data set.

        Args:
            bvh_folder (str): Bvh folder path.
            actors (list of str): List of actors to be included in the dataset.
            window (int, optional): Length of window. Defaults to 50.
            offset (int, optional): Offset of window. Defaults to 1.
            start_frame (int, optional):
                Override the start frame of each bvh file. Defaults to 0.
            device (str, optional): Device. e.g. "cpu", "cuda:0".
                Defaults to "cpu".
            dtype: torch.float16, torch.float32, torch.float64 etc.
        """
        super(BvhDataSet, self).__init__()
        self.bvh_folder = bvh_folder
        self.device = device
        self.dtype = dtype
        self.complete_flag=complete_flag
        self.augment=augment_flag
        self.names=[]
        
        if self.complete_flag:
            print("--only complete teeth--")
            self.bvh_folder=[os.path.join(bvh_folder,"complete")]
        else:
            self.bvh_folder=[os.path.join(bvh_folder,"complete"),os.path.join(bvh_folder,"incomplete")]
        # self.load_bvh_files()
        self.load_json_files()

    def _to_tensor(self, array):
        return torch.tensor(array, dtype=self.dtype, device=self.device)
    

    def load_json_files(self):
        # tooth data使用了json格式作为数据源
        self.bvh_files = []
        self.positions = []
        self.rotations = []
        self.data = []  # (28,14)
        self.target = []    # int
        self.pcd=[] # (28,2000,3)

        for dataset_path in self.bvh_folder:
            # load bvh files that match given actors
            for f in os.listdir(dataset_path):
                f = os.path.abspath(os.path.join(dataset_path, f))
                if f.endswith(".json"):
                    self.bvh_files.append(f)

        if not self.bvh_files:
            raise FileNotFoundError(
                "No bvh files found in {}.)".format(
                    self.bvh_folder)
            )

        self.bvh_files.sort()
        for bvh_path in self.bvh_files:
            print("Processing file {}".format(bvh_path))
            self.load_tooth_json(bvh_path)

        print("data count:{}".format(len(self.target)))


    def load_tooth_json(self,json_path):
            base,name=os.path.split(json_path)
            _, dirname=os.path.split(base)
            model_dir=os.path.join(r"E:\PROJECTS\tooth\Teeth_simulation_10K",dirname)
            meshes = self.load_teeth_stl(model_dir)
            meshes = np.asarray(meshes).reshape(28,2000,3)
            with open(json_path) as fh:
                data = json.load(fh)
                step_num = data["step_num"]
                positions = np.zeros((28,14), dtype=np.float32)
                temppos=np.zeros((step_num,28,7),dtype=np.float32)

                for step in data["steps"]:
                    step_id = step["step_id"]
                    teeth = step["tooth_data"]
                    for i in range(28):
                        temppos[step_id-1, i,:] = teeth[str(i)][:]
                positions[:,:7]=temppos[0,:,:]
                positions[:,7:]=temppos[step_num-1,:,:]
                self.data.append(self._to_tensor(positions))
                self.target.append(step_num)
                self.pcd.append(self._to_tensor(meshes))

                if self.augment:
                    for i in range(1,step_num-3):
                        positions[:,:7]=temppos[step_num-1-i,:,:]
                        positions[:,7:]=temppos[step_num-1,:,:]
                        self.data.append(self._to_tensor(positions))
                        self.target.append(step_num)
                        self.pcd.append(self._to_tensor(meshes))

    def load_teeth_stl(self,path):
        import struct
        ids = [i for i in range(17, 10, -1)] \
            + [i for i in range(21, 28)] \
            + [i for i in range(47, 40, -1)] \
            + [i for i in range(31, 38)]
        oid = {id: i for i, id in enumerate(ids)}   # dict{41:第0颗牙}
        # get .stl mesh
        teeth_mask = np.zeros(28, dtype=np.int32)
        gs = 0.3
        d = 0.05
        coords = []

        for id in ids:
            # for each tooth
            if not os.path.exists(f'{path}/models/{id}._Root.stl'):
                coords.append(None)
                continue
            with open(f'{path}/models/{id}._Root.stl', 'rb') as file:
                data = file.read()
            data = data[84:]
            s = b''
            ns = b''
            n = len(data) // 50 # face num:一个完整二进制STL文件的大小为三角形面片数乘以 50再加上84个字节。
            for i in range(n):
                s = s + data[50 * i + 12 : 50 * i + 48]# coords
                ns = ns + data[50*i:50*i+12]# normals
            data = struct.unpack('f' * n * 9, s)
            data = np.array(data, dtype=np.float32)
            data = np.reshape(data, (n * 3, 3))
            print(data.shape, end=' ')
            data = np.unique(data, axis=0)
            print(data.shape, end=' ')

            data=fixedNumDownSample(data,2000,1,0.1)
            coords.append(data)

        return coords


    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx],torch.transpose(self.pcd[idx],1,2),self.target[idx],idx)
        # data-pos :(28,14)
        # pcd:(28,3,2000)
