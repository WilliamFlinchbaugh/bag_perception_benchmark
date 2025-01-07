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

from glob import glob

from autoware_perception_msgs.msg import ObjectClassification
from autoware_perception_msgs.msg import Shape
from autoware_perception_msgs.msg import TrackedObject
from autoware_perception_msgs.msg import TrackedObjects
from geometry_msgs.msg import TransformStamped
from .benchmark_tools.math_utils import compose_transforms
import rclpy
from rclpy.time import Time
from rclpy.clock import ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf_transformations
# from unique_identifier_msgs.msg import UUID
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from pydoc import locate
from std_msgs.msg import String, Header
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from ament_index_python.packages import get_package_share_directory
import yaml
import os
import bisect


topic_filter_list = {
    "/sensing/imu/tamagawa/imu_raw",
    # "/sensing/lidar/left/pointcloud_raw_ex",
    # "/sensing/lidar/right/pointcloud_raw_ex",
    "/sensing/lidar/top/pointcloud_raw_ex",
    "/sensing/vehicle_velocity_converter/twist_with_covariance",
    "/sensing/imu/imu_data",
    "/map/vector_map",
    "/simulation/entity/marker",
    "/tf"
}

replay_topic_list = {x for x in topic_filter_list if "tf" not in x and "vector_map" not in x}

import_topic_list = {
    # "/sensing/lidar/left/pointcloud_raw_ex",
    # "/sensing/lidar/right/pointcloud_raw_ex",
    "/sensing/lidar/top/pointcloud_raw_ex",
}

class PlayerNode(Node):
    def __init__(self):
        super().__init__("bag_player_node")
        
        self.topic_filter_list = topic_filter_list
        self.replay_topic_list = replay_topic_list
        self.important_topics = import_topic_list
        
        # get the ros bag file path
        self.declare_parameter("bag_file", "")
        self.bag_file = self.get_parameter("bag_file").get_parameter_value().string_value
        
        # get the sensor model name
        self.declare_parameter("sensor_model", "")
        sensor_desc_pkg = self.get_parameter("sensor_model").get_parameter_value().string_value + "_description"
        self.sensor_tfs = self.get_sensor_tfs(sensor_desc_pkg)
        
        # read the bag file to get the topics
        self.reader = self.get_bag_reader()
        
        # break the dataset into frames that we can replay
        self.frames = self.get_frames_from_reader(self.reader)
        self.frame_idx = 0
        
        # create publishers for all topics
        self.publishers_ = {}
        for topic in self.replay_topic_list:
            if topic in self.typestr_to_type:
                # create a publisher for the topic
                self.publishers_[topic] = self.create_publisher(self.typestr_to_type[topic], topic, 1)
            else:
                self.get_logger().info(f"Topic {topic} not found in the bag file.")
        
        self.get_logger().info(f"Created publishers for {len(self.publishers_)} topics.")
        
        # create services
        self.srv_read_scene_data = self.create_service(
            Trigger, "read_current_segment", self.read_dataset_segment
        )
        self.srv_read_scene_data = self.create_service(
            Trigger, "send_frame", self.frame_processed_callback
        )
        self.srv_read_scene_data = self.create_service(
            Trigger, "reset_dataset", self.reset_dataset
        )
        self.pub_segment_finished = self.create_publisher(Bool, "segment_finished", 1)
        
        self.current_scene_processed = False
        
        # create the transform broadcaster for the tf and tf_static messages
        self.pose_broadcaster = TransformBroadcaster(self)
        self.static_tf_publisher = StaticTransformBroadcaster(self)
        
        # create the vector map publisher (make sure to set the QoS profile)
        qos_profile = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            depth=rclpy.qos.HistoryPolicy.KEEP_LAST,
            durability=rclpy.qos.DurabilityPolicy.TRANSIENT_LOCAL,
            liveliness=rclpy.qos.LivelinessPolicy.AUTOMATIC
        )
        self.vector_map_publisher = self.create_publisher(self.typestr_to_type["/map/vector_map"], "/map/vector_map", qos_profile)

    def get_bag_reader(self):
        self.get_logger().info("Reading bag file...")
        
        # init bag reader
        reader = SequentialReader()
        storage_options = StorageOptions(uri=self.bag_file, storage_id="sqlite3")
        converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
        reader.open(storage_options, converter_options)
        
        # get the bag metadata
        self.topic_types = reader.get_all_topics_and_types()
        
        # get the type string to type mapping
        self.typestr_to_type = {}
        for meta_info in self.topic_types:
            self.typestr_to_type[meta_info.name] = locate(meta_info.type.replace("/", "."))
        
        # filter to only the topics we want to replay
        reader.set_filter(StorageFilter(topics=list(self.topic_filter_list)))
        
        return reader
    
    def reset_dataset(self, request, response):
        self.get_logger().info("Resetting dataset...")
        self.reader = self.get_bag_reader()
        self.frames = self.get_frames_from_reader(self.reader)
        self.frame_idx = 0
        self.current_scene_processed = False
        self.pub_segment_finished.publish(Bool(data=False))
        response.success = True
        response.message = "Dataset reset."
        return response

    def frame_processed_callback(self, request, response):
        # publish next frame
        if self.frame_idx < len(self.frames):
            self.publish_next_frame()
            # self.get_logger().info("Frame published.")
            
            response.success = True
            response.message = "Frame published."
            return response
        
        # end of dataset
        else:
            self.get_logger().info("End of dataset!")
            msg = Bool()
            msg.data = True
            self.pub_segment_finished.publish(msg)

            response.success = False
            response.message = "Dataset finished."
            return response
    
    def publish_next_frame(self):
        # get the next frame
        frame = self.frames[self.frame_idx]
        
        # get frame info
        msgs = frame["msgs"]
        time = frame["time"]
        tfs = frame["tfs"]
        
        # publish the static tfs
        self.publish_static_tfs(time)
        
        # publish the vector map
        self.publish_vector_map(time)
        
        # publish the pose tfs
        for tf in tfs:
            self.pose_broadcaster.sendTransform(tf.transforms)
        
        # publish the messages in the frame
        for topic, msg in msgs:
            # self.get_logger().info(f"Publishing message on topic {topic}")
            self.publishers_[topic].publish(msg)
        
        self.frame_idx += 1
        
        
    def read_dataset_segment(self, request, response):
    #     if self.tf_segment_idx >= len(self.tf_list):
    #         self.get_logger().info("All Waymo segments in the given path have been processed.")
    #         exit()

    #     self.get_logger().info("Waymo segment decoding from dataset...")
    #     self.dataset = WaymoDataset(self.tf_list[self.tf_segment_idx])
    #     self.get_logger().info("Waymo segment decoded")
    #     self.tf_segment_idx += 1
        response.success = True
        response.message = "Segment read"
        return response
    
    def get_frames_from_reader(self, reader):
        
        sensor_data_msgs = {}
        tf_msgs = []
        self.vector_map_msg = None
        last_nanosec = None
        frame_idx = 0
        
        # skip until the first tf message
        while reader.has_next():
            topic, msg, _ = reader.read_next()
            if "vector_map" in topic:
                self.vector_map_msg = deserialize_message(msg, self.typestr_to_type[topic])
            elif "tf" in topic:
                msg = deserialize_message(msg, self.typestr_to_type[topic])
                tf_msgs.append((msg.transforms[0].header.stamp, topic, msg))
                break
        
        # read all messages
        while reader.has_next():
            topic, msg, _ = reader.read_next()
            msg = deserialize_message(msg, self.typestr_to_type[topic])
            if "tf" in topic:
                timestamp = msg.transforms[0].header.stamp
                tf_msgs.append((timestamp, topic, msg))
            
            elif "vector_map" in topic:
                self.vector_map_msg = msg
                
            else:
                if "marker" in topic:
                    timestamp = msg.markers[0].header.stamp
                else:
                    timestamp = msg.header.stamp
                
                if not last_nanosec:
                    last_nanosec = timestamp.nanosec
                    sensor_data_msgs[frame_idx] = []
                
                elif timestamp.nanosec != last_nanosec:
                    frame_idx += 1
                    sensor_data_msgs[frame_idx] = []
                
                sensor_data_msgs[frame_idx].append((timestamp, topic, msg))
                last_nanosec = timestamp.nanosec
        
        self.get_logger().info(f"Number of frames before filtering: {len(sensor_data_msgs)}")
        
        # filter out all of the frames that don't contain all of the replay topics
        filtered_frames = sensor_data_msgs.copy()
        for frame_idx, sensor_msgs in sensor_data_msgs.items():
            sensor_topics = {sensor_msg[1] for sensor_msg in sensor_msgs}
            if not self.important_topics.issubset(sensor_topics):
                del filtered_frames[frame_idx]
        
        self.get_logger().info(f"Number of frames after filtering: {len(filtered_frames)}")
        
        # reindex the frames after filtering
        sensor_data_msgs = {i: msgs for i, (_, msgs) in enumerate(filtered_frames.items())}
        
        # Sort TF messages by timestamp for efficient lookup
        tf_msgs.sort(key=lambda x: x[0].sec * 1e9 + x[0].nanosec)
        tf_timestamps = [tf[0].sec * 1e9 + tf[0].nanosec for tf in tf_msgs]

        frames = []
        
        for frame_idx, sensor_msgs in sensor_data_msgs.items():
            timestamp = sensor_msgs[0][0]
            sensor_nanosec = timestamp.sec * 1e9 + timestamp.nanosec
            
            # Find the closest TFs before and after the sensor message
            idx = bisect.bisect_left(tf_timestamps, sensor_nanosec)
            
            # TFs before the sensor message
            tfs_before = [tf_msgs[i] for i in range(idx) if tf_msgs[i][0].sec * 1e9 + tf_msgs[i][0].nanosec <= sensor_nanosec]
            
            # TFs after the sensor message
            tfs_after = [tf_msgs[i] for i in range(idx, len(tf_msgs)) if tf_msgs[i][0].sec * 1e9 + tf_msgs[i][0].nanosec >= sensor_nanosec]
            
            frame_tfs = tfs_before[-2:] + tfs_after[:2]
            
            # get the transforms
            frame_tfs = [tf[2] for tf in frame_tfs]
            
            # skip if we don't have a tf before and after
            if len(frame_tfs) != 4:
                continue
            
            # Create the frame with the sensor message, the closest TFs before and after
            frame = {
                'msgs': [(sensor_msg[1], sensor_msg[2]) for sensor_msg in sensor_msgs],
                'tfs': frame_tfs,
                'time': timestamp,
            }
            frames.append(frame)
        
        self.get_logger().info(f"Vector map exists: {self.vector_map_msg != None}")
    
        return frames
    
    def get_sensor_tfs(self, sensor_desc_name):
        # get the paths to the sensor model yaml files
        sensor_desc_path = get_package_share_directory(sensor_desc_name)
        sensor_kit_calibration_path = os.path.join(sensor_desc_path, "config", f"sensor_kit_calibration.yaml")
        sensors_calibration_path = os.path.join(sensor_desc_path, "config", f"sensors_calibration.yaml")
        
        # load the sensor calibration yaml files
        sensor_kit_calibration = yaml.safe_load(open(sensor_kit_calibration_path, "r"))
        sensors_calibration = yaml.safe_load(open(sensors_calibration_path, "r"))
        
        # we need to publish a tf from base_link to velodyne_front_base_link
        # so we need to get the transform from base_link to sensor_kit_base_link
        # and the transform from sensor_kit_base_link to velodyne_front_base_link
        # and the we combine them to get the transform from base_link to velodyne_front_base_link
        base_link_to_sensor_kit = self.get_transform_from_dict(sensors_calibration["base_link"]["sensor_kit_base_link"])
        sensor_kit_to_sensor = {}
        for sensor, tf_dict in sensor_kit_calibration["sensor_kit_base_link"].items():
            sensor_kit_to_sensor[sensor] = self.get_transform_from_dict(tf_dict)
        
        # now we need to combine the transforms
        sensor_tfs = {}
        for sensor, tf in sensor_kit_to_sensor.items():
            sensor_tfs[sensor] = compose_transforms(base_link_to_sensor_kit, tf)
        
        # convert to TransformStamped
        for sensor, tf in sensor_tfs.items():
            ts = TransformStamped()
            ts.header.frame_id = "base_link"
            
            # update the sensor name
            sensor_name = sensor
            if "_base_link" in sensor:
                sensor_name = sensor.replace("_base_link", "")
                
            ts.child_frame_id = sensor_name
            ts.transform = tf
            sensor_tfs[sensor] = ts
        
        return sensor_tfs
        
    def get_transform_from_dict(self, tf_dict):
        t = Transform()
        t.translation = Vector3(x=tf_dict["x"], y=tf_dict["y"], z=tf_dict["z"])
        q = tf_transformations.quaternion_from_euler(tf_dict["roll"], tf_dict["pitch"], tf_dict["yaw"])
        t.rotation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        return t
    
    def publish_static_tfs(self, time):
        for tf in self.sensor_tfs.values():
            tf.header.stamp = time
        self.static_tf_publisher.sendTransform(list(self.sensor_tfs.values()))
    
    def publish_vector_map(self, time):
        if self.vector_map_msg:
            # self.get_logger().info("Publishing vector map...")
            self.vector_map_msg.header.stamp = time
            self.vector_map_publisher.publish(self.vector_map_msg)
    
    def scene_processed(self):
        return self.current_scene_processed

    def set_scene_processed(self, value):
        self.current_scene_processed = value


def main(args=None):
    rclpy.init(args=args)
    perception_benchmark = PlayerNode()
    rclpy.spin(perception_benchmark)
    perception_benchmark.destroy_node()
    rclpy.shutdown()
