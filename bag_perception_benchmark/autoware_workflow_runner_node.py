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
from autoware_perception_msgs.msg import DetectedObjects
import rclpy.qos
from sensor_msgs.msg import PointCloud2
import psutil
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import threading
from traffic_simulator_msgs.msg import EntityStatusWithTrajectoryArray

capture_topics = {
    "/perception/object_recognition/detection/centerpoint/objects"
}

class RunnerNode(Node):
    def __init__(self):
        super().__init__("autoware_workflow_runner_node")

        self.declare_parameter("launch_file", "")
        self.launch_file = self.get_parameter("launch_file").get_parameter_value().string_value

        self.declare_parameter("vehicle_model", "")
        self.vehicle_model = self.get_parameter("vehicle_model").get_parameter_value().string_value

        self.declare_parameter("sensor_model", "")
        self.sensor_model = self.get_parameter("sensor_model").get_parameter_value().string_value

        self.autoware_pid = None
        self.timer_subs_checker = None

        self.client_read_dataset_futures = []
        self.client_read_dataset = self.create_client(Trigger, "read_current_segment")
        while not self.client_read_dataset.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("service not available, waiting again...")

        self.client_read_frame_futures = []
        self.client_read_dataset_frame = self.create_client(Trigger, "send_frame")
        while not self.client_read_dataset_frame.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("service not available, waiting again...")
        
        self.client_reset_dataset_futures = []
        self.client_reset_dataset = self.create_client(Trigger, "reset_dataset")
        while not self.client_reset_dataset.wait_for_service(timeout_sec=3.0):
            self.get_logger().info("service not available, waiting again...")

        self.sub_segment_finished = self.create_subscription(
            Bool, "segment_finished", self.segment_finished_callback, 1
        )
        
        self.outputs_to_save = {}
        
        # setup subscriber to detected objects from centerpoint
        self.det_objects_frames = []
        self.sub_det_objects = self.create_subscription(
            DetectedObjects, "/perception/object_recognition/detection/centerpoint/objects", self.det_objects_callback, 1
        )
        
        # setup subscriber to ground truth objects
        self.gt_objects_frames = []
        self.sub_gt_objects = self.create_subscription(
            EntityStatusWithTrajectoryArray, "/simulation/entity/status", self.gt_objects_callback, 1
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
        
        # save the detected objects to a json file
        self.save_outputs()
        
        # kill autoware
        self.kill_autoware(self.autoware_pid)
        
        # uncomment to have the dataset reset after each segment
        req = Trigger.Request()
        self.client_reset_dataset_futures.append(self.client_reset_dataset.call_async(req))
        
        self.read_dataset_request()

    def wait_until_autoware_subs_ready(self):

        self.get_logger().info("Waiting for Autoware's subscriber to be ready")

        if self.check_lidar_model_ready():
            self.get_logger().info("Autoware ready.")
            self.read_frame_request()
            self.destroy_timer(self.timer_subs_checker)

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

    def det_objects_callback(self, det_objects):
        # num_objects = len(det_objects.objects)
        # self.get_logger().info(f"Receieved object detection")
        # self.det_objects_frames.append(det_objects)
        self.read_frame_request()
        
    def gt_objects_callback(self, gt_objects):
        self.gt_objects_frames.append(gt_objects)
        
    def save_outputs(self):
        pass
        

def main(args=None):
    rclpy.init(args=args)
    autoware_workflow_runner_node = RunnerNode()
    autoware_workflow_runner_node.spin()
    rclpy.shutdown()
