from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
import open3d as o3d
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from pydoc import locate
import matplotlib.pyplot as plt


# To use:
# 1. Set the bag_attack and bag_noattack variables to the paths of the bags you want to compare

# 2. Set the pointcloud_idx variable to the index of the pointcloud message you want to visualize

# 3. Set the visualization parameters (zoom, front, lookat, up) to the desired view of the pointcloud
# For this, you can use the Open3D visualizer when the script runs, manipulate the view to get the desired view, then
# press Ctrl + C to copy the view parameters as JSON, then just copy the values into the script

# 4. Run the script, press "A" twice to run the callback functions that save the depth images
# The depth images will be saved as "attack_depth.png" and "noattack_depth.png" in the current directory
# When you close the first visualization window, the second one will open automatically


# Bags to compare

# PRA
# bag_attack = "/media/william/Extreme SSD/lidar_data/PRA_vehicles_20_stereo_scenario_1.db3"
# bag_noattack = "/home/william/lidar_data/stereo_scenario_1.db3"

# ORA
bag_attack = "/home/william/lidar_data/ORA_Pedestrians/ORA_pedestrians_50_3m_stereo_scenario_1.db3"
bag_noattack = "/home/william/lidar_data/stereo_scenario_1.db3"

# pick out the same pointcloud in both bags
pointcloud_idx = 420
pointcloud_topic = "/sensing/lidar/top/pointcloud_raw_ex"
attack_pc_msg = None
noattack_pc_msg = None

# visualization parameters
# PRA settings
# zoom = 0.035
# front = [ -0.15016850725616354, 0.89604269384592894, 0.41780008405191427 ]
# lookat = [ 0.51183743823490568, -9.579856584182707, 0.7888615655843737 ]
# up = [ 0.087807463872090513, -0.408834364168844, 0.90837454387659311 ]

# ORA settings
zoom = 0.02
front = [ 0.0013606193497739856, 0.87082159485096033, 0.49159729317421497 ]
lookat = [ 4.0173923876410624, -8.2932052206102966, -0.24134183676236182 ]
up = [ -0.072182995040332057, -0.49022983986166446, 0.86859905556949246 ]


# read the bag with attack
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_attack, storage_id="sqlite3")
converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
reader.open(storage_options, converter_options)

# filter to only read the topics we care about
filter = StorageFilter(topics=[pointcloud_topic])
reader.set_filter(filter)

# get the bag metadata
topic_types = reader.get_all_topics_and_types()

# get the type string to type mapping
typestr_to_type = {}
for meta_info in topic_types:
    typestr_to_type[meta_info.name] = locate(meta_info.type.replace("/", "."))

# read the messages
idx = 0
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == pointcloud_topic:
        if idx == pointcloud_idx:
            attack_pc_msg = msg
            break
        idx += 1

if attack_pc_msg is None:
    raise ValueError(f"Could not find pointcloud message at index {pointcloud_idx} in bag {bag_attack}")

# read the bag with no attack
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_noattack, storage_id="sqlite3")
converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
reader.open(storage_options, converter_options)

# filter to only read the topics we care about
filter = StorageFilter(topics=[pointcloud_topic])
reader.set_filter(filter)

# read the messages
idx = 0
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == pointcloud_topic and idx == pointcloud_idx:
        noattack_pc_msg = msg
        break
    idx += 1

if noattack_pc_msg is None:
    raise ValueError(f"Could not find pointcloud message at index {pointcloud_idx} in bag {bag_noattack}")

# convert the pointcloud messages to open3d pointclouds
attack_pc = pc2.read_points(attack_pc_msg, field_names=("x", "y", "z"), skip_nans=True)
attack_pc_np = np.array([[point[0], point[1], point[2]] for point in attack_pc], dtype=np.float32)
attack_pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(attack_pc_np))

noattack_pc = pc2.read_points(noattack_pc_msg, field_names=("x", "y", "z"), skip_nans=True)
noattack_pc_np = np.array([[point[0], point[1], point[2]] for point in noattack_pc], dtype=np.float32)
noattack_pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(noattack_pc_np))

# visualize the pointclouds by saving the depth images




def anim_callback_attack(vis):
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    depth = vis.capture_depth_float_buffer()
    
    # create image from depth buffer
    plt.imsave("attack_depth.png", np.asarray(depth))
    return False

def anim_callback_noattack(vis):
    ctr = vis.get_view_control()
    ctr.set_zoom(zoom)
    ctr.set_front(front)
    ctr.set_lookat(lookat)
    ctr.set_up(up)
    
    depth = vis.capture_depth_float_buffer()
    
    # create image from depth buffer
    plt.imsave("noattack_depth.png", np.asarray(depth))
    return False

key_to_callback = {
    ord("A"): anim_callback_noattack,
}

o3d.visualization.draw_geometries_with_key_callbacks([noattack_pc_o3d], key_to_callback=key_to_callback)

key_to_callback = {
    ord("A"): anim_callback_attack,
}

o3d.visualization.draw_geometries_with_key_callbacks([attack_pc_o3d], key_to_callback=key_to_callback)

