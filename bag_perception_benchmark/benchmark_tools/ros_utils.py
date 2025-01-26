# Copyright 2018 Autoware Foundation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import TransformStamped
import numpy as np
from bag_perception_benchmark.benchmark_tools.math_utils import build_affine
from bag_perception_benchmark.benchmark_tools.math_utils import decompose_affine
from bag_perception_benchmark.benchmark_tools.math_utils import rotation_matrix_to_euler_angles
from bag_perception_benchmark.benchmark_tools.math_utils import transform_to_affine
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
import tf_transformations
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header


def create_empty_point_cloud_msg(header):
    dtype = np.dtype([
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.uint8),
        ('return_type', np.uint8),
        ('channel', np.uint16),
        ('azimuth', np.float32),
        ('elevation', np.float32),
        ('distance', np.float32),
        ('time_stamp', np.uint32)
    ])
    fields = [
        PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=pc2.PointField.UINT8, count=1),
        PointField(name='return_type', offset=13, datatype=pc2.PointField.UINT8, count=1),
        PointField(name='channel', offset=14, datatype=pc2.PointField.UINT16, count=1),
        PointField(name='azimuth', offset=16, datatype=pc2.PointField.FLOAT32, count=1),
        PointField(name='elevation', offset=20, datatype=pc2.PointField.FLOAT32, count=1),
        PointField(name='distance', offset=24, datatype=pc2.PointField.FLOAT32, count=1),
        PointField(name='time_stamp', offset=28, datatype=pc2.PointField.UINT32, count=1)
    ]
    
    structured_pc = np.zeros(1, dtype=dtype)
    pc = pc2.create_cloud(header=header, fields=fields, points=structured_pc)
    return pc
