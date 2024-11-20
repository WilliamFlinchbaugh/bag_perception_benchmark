from glob import glob
import os

from setuptools import setup
from setuptools import find_packages

package_name = "bag_perception_benchmark"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=['test']),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.*")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="William Flinchbaugh",
    maintainer_email="williamflinchbaugh@gmail.com",
    description="Tool to evaluate the autoware perception stack on recorded rosbag datasets",
    license="Apache License 2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "benchmark_node = bag_perception_benchmark.benchmark_node:main",
            "bag_player_node = bag_perception_benchmark.bag_player_node:main",
            "autoware_workflow_runner_node = bag_perception_benchmark.autoware_workflow_runner_node:main",
        ],
    },
)
