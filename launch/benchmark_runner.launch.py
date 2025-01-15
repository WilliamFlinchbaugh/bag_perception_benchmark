import os
from os.path import join as joinPath

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    dataset_path_launch_arg = DeclareLaunchArgument(
        "bag_file",
        default_value=joinPath(os.environ["HOME"], "bag_perception_benchmark", "sample.bag"),
        description="rosbag file to replay (.db3)",
    )

    launch_file_launch_arg = DeclareLaunchArgument(
        "launch_file",
        default_value="benchmark_perception.launch.xml",
        description="Launch file for testing perception stack",
    )

    vehicle_model_launch_arg = DeclareLaunchArgument(
        "vehicle_model",
        default_value="awsim_labs_vehicle",
        description="",
    )

    sensor_model_launch_arg = DeclareLaunchArgument(
        "sensor_model",
        default_value="awsim_labs_sensor_kit",
        description="",
    )
    
    top_lidar_only_launch_arg = DeclareLaunchArgument(
        "top_lidar_only",
        default_value="False",
        description="",
    )

    bag_player_node = Node(
        package="bag_perception_benchmark",
        name="bag_player_node",
        executable="bag_player_node",
        output="screen",
        parameters=[
            {
                "bag_file": LaunchConfiguration("bag_file"),
                "sensor_model": LaunchConfiguration("sensor_model"),
                "top_lidar_only": LaunchConfiguration("top_lidar_only"),
            }
        ],
    )

    autoware_workflow_runner_node = Node(
        package="bag_perception_benchmark",
        name="autoware_workflow_runner_node",
        executable="autoware_workflow_runner_node",
        output="screen",
        parameters=[
            {
                "launch_file": LaunchConfiguration("launch_file"),
                "vehicle_model": LaunchConfiguration("vehicle_model"),
                "sensor_model": LaunchConfiguration("sensor_model"),
                "bag_file": LaunchConfiguration("bag_file"),
            }
        ],
    )

    return LaunchDescription(
        [
            dataset_path_launch_arg,
            launch_file_launch_arg,
            vehicle_model_launch_arg,
            sensor_model_launch_arg,
            top_lidar_only_launch_arg,
            bag_player_node,
            autoware_workflow_runner_node,
        ]
    )
