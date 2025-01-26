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

from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Transform
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Vector3
from .benchmark_tools.math_utils import compose_transforms
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
import tf_transformations
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from pydoc import locate
from ament_index_python.packages import get_package_share_directory
import yaml
import os
import pandas as pd
import re
import time


topic_filter_list = {
    "/sensing/imu/tamagawa/imu_raw",
    "/sensing/vehicle_velocity_converter/twist_with_covariance",
    "/sensing/imu/imu_data",
    "/map/vector_map",
    "/simulation/entity/marker",
    "/simulation/entity/status",
    "/tf"
}

replay_topic_list = {x for x in topic_filter_list if "tf" not in x and "vector_map" not in x}

class PlayerNode(Node):
    def __init__(self):
        super().__init__("bag_player_node")
        
        # get the ros bag file path
        self.declare_parameter("bag_file", "")
        self.bag_file = self.get_parameter("bag_file").get_parameter_value().string_value
        
        # get the sensor model name
        self.declare_parameter("sensor_model", "")
        sensor_kit_name = self.get_parameter("sensor_model").get_parameter_value().string_value
        sensor_launch_pkg = sensor_kit_name + "_launch"
        self.important_topics = self.get_lidar_topics_from_sensor_kit(sensor_launch_pkg)
        sensor_desc_pkg = sensor_kit_name + "_description"
        self.sensor_tfs = self.get_sensor_tfs(sensor_desc_pkg)
        
        # get if we should only use the top lidar sensor
        self.declare_parameter("top_lidar_only", False)
        self.top_lidar_only = self.get_parameter("top_lidar_only").get_parameter_value().bool_value
        if self.top_lidar_only:
            self.important_topics = {"/sensing/lidar/top/pointcloud_raw_ex"}
            self.get_logger().info("Only using top lidar sensor, topics: /sensing/lidar/top/pointcloud_raw_ex")
        
        self.topic_filter_list = topic_filter_list
        self.replay_topic_list = replay_topic_list
        
        self.topic_filter_list.update(self.important_topics)
        self.replay_topic_list.update(self.important_topics)
        
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
            if meta_info.name in self.topic_filter_list:
                self.get_logger().info(f"Found topic {meta_info.name} of type {meta_info.type}")
        
        # filter to only the topics we want to replay
        reader.set_filter(StorageFilter(topics=list(self.topic_filter_list)))
        
        return reader
    
    def frame_processed_callback(self, request, response):
        # publish next frame
        if self.frame_idx < len(self.frames):
            self.publish_next_frame()
            
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
        
        # publish static tfs and vector map
        self.publish_static_tfs()
        self.publish_vector_map()
        
        # after publishing tfs, wait 0.01 seconds
        
        for msg in frame:
            topic = msg["topic"]
            message = msg["msg"]
            
            # if this is a tf message, publish it with current timestamp
            if topic == "/tf":
                for tf in message.transforms:
                    self.publish_tf(tf)
                time.sleep(0.01)
            
            # for other messages, set the timestamp to now and publish
            else:
                message = self.set_timestamp_to_now(message, topic)
                self.publishers_[topic].publish(message)
            
            # self.get_logger().info(f"Published {topic}")
        
        self.frame_idx += 1
    
    def set_timestamp_to_now(self, msg, topic):
        if "/simulation/entity/marker" in topic:
            for marker in msg.markers:
                marker.header.stamp = self.get_clock().now().to_msg()
        elif "/simulation/entity/status" in topic:
            for entity in msg.data:
                entity.time = float(self.get_clock().now().nanoseconds)
        else:
            msg.header.stamp = self.get_clock().now().to_msg()
        return msg
        
    def read_dataset_segment(self, request, response):
        response.success = True
        response.message = "Segment read"
        return response
    
    def get_frames_from_reader(self, reader):
        self.get_logger().info("Getting frames from reader...")
        # advance the reader until the first /tf message (and get the /map/vector_map message)
        while reader.has_next():
            topic, msg, t = reader.read_next()
            if topic == "/map/vector_map":
                self.vector_map_msg = deserialize_message(msg, self.typestr_to_type[topic])
                continue
            if topic == "/tf":
                break
        
        msgs_dict = {
            "topic": [],
            "msg": [],
            "abs_time": [],
            "time_diff": []
        }
        
        # add the first tf message to the list
        msgs_dict["topic"].append(topic)
        msgs_dict["msg"].append(deserialize_message(msg, self.typestr_to_type[topic]))
        msgs_dict["abs_time"].append(t)
        msgs_dict["time_diff"].append(0)
        
        while reader.has_next():
            topic, msg, t = reader.read_next()
            msg = deserialize_message(msg, self.typestr_to_type[topic])
            msgs_dict["topic"].append(topic)
            msgs_dict["msg"].append(msg)
            
            # get time elapsed since last message in seconds
            time_diff = (t - msgs_dict["abs_time"][-1]) / 1e9
            msgs_dict["abs_time"].append(t)
            msgs_dict["time_diff"].append(time_diff)
        
        # convert to dataframe
        df = pd.DataFrame(msgs_dict)
        
        # split the dataframe into frames where each frame ends with a /tf message and contains all the important topics
        frames = []
        frame = []
        important_seen = set()
        for _, row in df.iterrows():
            topic = row["topic"]
            if topic in self.important_topics:
                important_seen.add(topic)
            
            frame.append(row)
            
            if topic == "/tf":
                if important_seen == self.important_topics:
                    frames.append(frame)
                    frame = []
                    important_seen = set()
        
        self.get_logger().info(f"Got {len(frames)} frames.")
        
        # for debugging, create a text file with the topics in each frame, and a newline between frames
        # with open("frames.txt", "w") as f:
        #     for frame in frames:
        #         for msg in frame:
        #             f.write(msg["topic"] + "\n")
        #         f.write("\n")
        
        # self.get_logger().info("Frames saved to frames.txt.")
        
        return frames
    
    def get_lidar_topics_from_sensor_kit(self, sensor_launch_pkg):
        # get the path to the pointcloud_preprocessor python file
        sensor_launch_path = get_package_share_directory(sensor_launch_pkg)
        pc_preprocessor_path = os.path.join(sensor_launch_path, "launch", "pointcloud_preprocessor.launch.py")
        
        pointcloud_topic_pattern = "/sensing/lidar/[a-z0-9]+/pointcloud_before_sync"
        
        # capture the lidar topics from the pointcloud_preprocessor launch file
        lidar_topics = set()
        with open(pc_preprocessor_path, "r") as f:
            for line in f:
                matches = re.findall(pointcloud_topic_pattern, line)
                for match in matches:
                    lidar_topics.add(match)

        # replace "pointcloud_before_sync" with "pointcloud_raw_ex"
        lidar_topics = {topic.replace("pointcloud_before_sync", "pointcloud_raw_ex") for topic in lidar_topics}
        
        self.get_logger().info(f"Found {len(lidar_topics)} lidar topics for sensor kit {sensor_launch_pkg}.")
        self.get_logger().info(', '.join(lidar_topics))
        
        return lidar_topics
    
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
    
    def publish_static_tfs(self):
        for tf in self.sensor_tfs.values():
            tf.header.stamp = self.get_clock().now().to_msg()
        self.static_tf_publisher.sendTransform(list(self.sensor_tfs.values()))
    
    def publish_vector_map(self):
        if self.vector_map_msg:
            # self.get_logger().info("Publishing vector map...")
            self.vector_map_msg.header.stamp = self.get_clock().now().to_msg()
            self.vector_map_publisher.publish(self.vector_map_msg)
    
    def publish_tf(self, tf):
        tf.header.stamp = self.get_clock().now().to_msg()
        self.pose_broadcaster.sendTransform(tf)
    
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
