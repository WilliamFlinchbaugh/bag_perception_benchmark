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

import signal
from subprocess import Popen, STDOUT, PIPE, DEVNULL
from autoware_perception_msgs.msg import DetectedObjects, TrackedObjects, PredictedObjects
from traffic_simulator_msgs.msg import EntityStatusWithTrajectoryArray
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import PointCloud2
import psutil
import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.serialization import serialize_message
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import threading
from visualization_msgs.msg import MarkerArray
import rosbag2_py
from rosbag2_py._storage import TopicMetadata
import os
import shutil

capture_topics = {
    "/perception/object_recognition/detection/centerpoint/objects": DetectedObjects,
    "/perception/object_recognition/detection/centerpoint/validation/objects": DetectedObjects,
    "/perception/object_recognition/detection/objects": DetectedObjects,
    "/perception/object_recognition/detection/clustering/objects": DetectedObjects,
    "/perception/object_recognition/objects": PredictedObjects,
    "/perception/object_recognition/detection/detection_by_tracker/objects": DetectedObjects,
    "/perception/object_recognition/tracking/objects": TrackedObjects,
    "/simulation/entity/marker": MarkerArray,
    "/simulation/entity/status": EntityStatusWithTrajectoryArray,
    "/tf": TFMessage,
}

msg_to_str = {
    DetectedObjects: "autoware_perception_msgs/msg/DetectedObjects",
    TrackedObjects: "autoware_perception_msgs/msg/TrackedObjects",
    PredictedObjects: "autoware_perception_msgs/msg/PredictedObjects",
    EntityStatusWithTrajectoryArray: "traffic_simulator_msgs/msg/EntityStatusWithTrajectoryArray",
    MarkerArray: "visualization_msgs/msg/MarkerArray",
    TFMessage: "tf2_msgs/msg/TFMessage",
}

final_topic = "/perception/object_recognition/objects"

class RunnerNode(Node):
    def __init__(self):
        super().__init__("autoware_workflow_runner_node")
        
        self.capture_topics = capture_topics
        self.final_topic = final_topic

        self.declare_parameter("launch_file", "")
        self.launch_file = self.get_parameter("launch_file").get_parameter_value().string_value

        self.declare_parameter("vehicle_model", "")
        self.vehicle_model = self.get_parameter("vehicle_model").get_parameter_value().string_value

        self.declare_parameter("sensor_model", "")
        self.sensor_model = self.get_parameter("sensor_model").get_parameter_value().string_value
        
        self.autoware_pid = None
        self.timer_subs_checker = None
        
        self.declare_parameter("bag_file", "")
        bag_file = self.get_parameter("bag_file").get_parameter_value().string_value
        
        # get output bag file path
        output_bag_path = os.path.join(os.path.dirname(bag_file), os.path.basename(bag_file).replace(".db3", "_output"))
        if os.path.exists(output_bag_path):
            shutil.rmtree(output_bag_path)
        
        # open the output bag
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(uri=output_bag_path, storage_id="sqlite3")
        converter_options = rosbag2_py.ConverterOptions('', '')
        self.writer.open(storage_options, converter_options)
        
        self.client_read_dataset_futures = []
        self.client_read_dataset = self.create_client(Trigger, "read_current_segment")
        while not self.client_read_dataset.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("service not available, waiting again...")

        self.client_read_frame_futures = []
        self.client_read_dataset_frame = self.create_client(Trigger, "send_frame")
        while not self.client_read_dataset_frame.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("service not available, waiting again...")

        self.sub_segment_finished = self.create_subscription(
            Bool, "segment_finished", self.segment_finished_callback, 1
        )
        
        qos_profile = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            liveliness=rclpy.qos.LivelinessPolicy.AUTOMATIC
        )
        self.sub_final_topic = self.create_subscription(
            self.capture_topics[self.final_topic], self.final_topic, self.final_topic_callback, qos_profile
        )
        
        self.read_dataset_request()

    def spin(self):

        while rclpy.ok():
            rclpy.spin_once(self)

            incomplete_read_dataset_futures = []
            for f in self.client_read_dataset_futures:
                if f.done():
                    response = f.result()
                    if response.success:
                        self.autoware_pid = self.run_autoware()
                        self.timer_subs_checker = self.create_timer(
                            2, self.wait_until_autoware_subs_ready
                        )
                else:
                    incomplete_read_dataset_futures.append(f)

            self.client_read_dataset_futures = incomplete_read_dataset_futures

            incomplete_send_frame_futures = []
            for f in self.client_read_frame_futures:
                if f.done():
                    response = f.result()
                else:
                    incomplete_send_frame_futures.append(f)

            self.client_read_frame_futures = incomplete_send_frame_futures

    def read_dataset_request(self):
        req = Trigger.Request()
        self.client_read_dataset_futures.append(self.client_read_dataset.call_async(req))

    def read_frame_request(self):
        req = Trigger.Request()
        self.client_read_frame_futures.append(self.client_read_dataset_frame.call_async(req))

    def segment_finished_callback(self, ready):
        self.get_logger().info("Autoware is being killed. ")
        
        # kill autoware
        self.kill_autoware(self.autoware_pid)
        
        # run benchmark node?
        

    def wait_until_autoware_subs_ready(self):

        self.get_logger().info("Waiting for Autoware's subscriber to be ready")

        if self.check_lidar_model_ready():
            self.get_logger().info("Autoware ready.")
            self.setup_output_bag()
            self.read_frame_request()
            self.destroy_timer(self.timer_subs_checker)
            
    def setup_output_bag(self):
        # create subscriber for each topic
        self.subscribers = {}
        qos_profile = QoSProfile(
            reliability=rclpy.qos.ReliabilityPolicy.RELIABLE,
            history=rclpy.qos.HistoryPolicy.KEEP_LAST,
            depth=10,
            durability=rclpy.qos.DurabilityPolicy.VOLATILE,
            liveliness=rclpy.qos.LivelinessPolicy.AUTOMATIC
        )
        
        self.callbacks = {}
        
        for idx, (topic, msg_type) in enumerate(self.capture_topics.items()):
            # create topic in bag
            type_str = msg_to_str[msg_type]
            topic_info = rosbag2_py._storage.TopicMetadata(
                name=topic,
                type=type_str,
                serialization_format='cdr'
            )
            self.writer.create_topic(topic_info)
            
            # setup callback for each topic
            self.callbacks[topic] = lambda msg, topic=topic: self.write_to_bag(msg, topic)
            
            # create subscriber            
            self.subscribers[topic] = self.create_subscription(
                msg_type, topic, self.callbacks[topic], qos_profile
            )
    
    def write_to_bag(self, msg, topic):
        serialized_msg = serialize_message(msg)
        self.writer.write(topic, serialized_msg, self.get_clock().now().nanoseconds)

    def run_autoware(self):
        cmd = (
            "ros2 launch bag_perception_benchmark "
            + self.launch_file
            + " vehicle_model:="
            + self.vehicle_model
            + " sensor_model:="
            + self.sensor_model
            + " rviz:=false"
        )
        
        self.get_logger().info("Running Autoware with command: " + cmd)
        
        launch_process = Popen(cmd, text=False, shell=True, stdout=PIPE)
        
        # asynchoronously read the output of the process
        self.print_thread = threading.Thread(target=self.print_autoware_output, args=(launch_process,))
        self.print_thread.start()
        
        return launch_process.pid
    
    def print_autoware_output(self, launch_process):
        while True:
            output = launch_process.stdout.readline()
            if output == "" and launch_process.poll() is not None:
                break
            if output:
                self.get_logger().info(output.decode("utf-8"))

    def kill_autoware(self, parent_pid, sig=signal.SIGTERM):
        try:
            parent = psutil.Process(parent_pid)
        except psutil.NoSuchProcess:
            return
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(sig)
        parent.send_signal(sig)
        
        if self.print_thread.is_alive():
            self.print_thread.join()

    def check_lidar_model_ready(self):
        centerpoint_ready = self.count_publishers(
            "/perception/object_recognition/detection/centerpoint/objects"
        )
        apollo_ready = self.count_publishers(
            "/perception/object_recognition/detection/apollo/labeled_clusters"
        )
        return bool(centerpoint_ready or apollo_ready)

    def final_topic_callback(self, det_objects):
        self.read_frame_request()
        

def main(args=None):
    try:
        rclpy.init(args=args)
        autoware_workflow_runner_node = RunnerNode()
        autoware_workflow_runner_node.spin()
        rclpy.shutdown()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
        
