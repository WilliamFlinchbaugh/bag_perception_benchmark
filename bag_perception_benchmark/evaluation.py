from autoware_perception_msgs.msg import PredictedObjects, PredictedObject, PredictedObjectKinematics, ObjectClassification, Shape
from traffic_simulator_msgs.msg import EntityStatusWithTrajectoryArray, EntityStatus, BoundingBox
from geometry_msgs.msg import Point, Vector3
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from pydoc import locate
import argparse
import numpy as np
from benchmark_tools.eval_utils import DetectionObject, calculate_metrics


# topic to read from for the predicted objects
predicted_objs_topic = "/perception/object_recognition/objects"

# topic to read from for the ground truth objects
gt_topic = "/simulation/entity/status"


# get the bag file
argparser = argparse.ArgumentParser()
argparser.add_argument("--bag_file", type=str, required=True)
args = argparser.parse_args()
bag_file = args.bag_file

# read the bag
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_file, storage_id="sqlite3")
converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
reader.open(storage_options, converter_options)

# filter to only read the topics we care about
filter = StorageFilter(topics=[predicted_objs_topic, gt_topic, "/tf"])
reader.set_filter(filter)

# get the bag metadata
topic_types = reader.get_all_topics_and_types()

# get the type string to type mapping
typestr_to_type = {}
for meta_info in topic_types:
    typestr_to_type[meta_info.name] = locate(meta_info.type.replace("/", "."))

# read until we get the first predicted objects message
predicted_objects_msgs = []
gt_objects_msgs = []
car_positions = []
objects_seen = set()
first_distances = []
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == gt_topic:
        gt_objects_msgs.append(msg)
    
    elif topic == "/tf":
        # save the current position of the base link for later use
        for transform in msg.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "base_link":
                car_positions.append(transform.transform.translation)
    
    elif topic == predicted_objs_topic:
        # only consider messages with objects detected
        if len(msg.objects) == 0:
            continue
        # get uuid as string and add to the set of objects seen
        uuid_str = "".join([f"{b:02x}" for b in msg.objects[0].object_id.uuid])
        objects_seen.add(uuid_str)
        
        # append the message to the list
        # also keep track of the index of the last gt message and the car's last position for later matching
        predicted_objects_msgs.append((msg, len(gt_objects_msgs)-1, car_positions[-1]))
        break

# after we get the first predicted objects message, read the rest of the messages
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == gt_topic:
        gt_objects_msgs.append(msg)
    
    elif topic == "/tf":
        # save the current position of the base link for later use
        for transform in msg.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "base_link":
                car_positions.append(transform.transform.translation)
    
    elif topic == predicted_objs_topic:
        for obj in msg.objects:
            uuid_str = "".join([f"{b:02x}" for b in obj.object_id.uuid])
            if uuid_str not in objects_seen:
                first_distances.append(np.linalg.norm([car_positions[-1].x - obj.kinematics.initial_pose_with_covariance.pose.position.x, car_positions[-1].y - obj.kinematics.initial_pose_with_covariance.pose.position.y]))
                objects_seen.add(uuid_str)
        
        # append the message to the list
        # also keep track of the index of the last gt message and the car's last position for later matching
        predicted_objects_msgs.append((msg, len(gt_objects_msgs)-1, car_positions[-1]))

# range to consider a ground truth object as able to be detected
# if within this range and not detected, it is a false negative
detection_range = 20

# convert the messages to DetectionObject objects
frames = []
for pred_objs, gt_idx, car_pos in predicted_objects_msgs:
    pred_objs = [DetectionObject().init_with_pred(pred_obj) for pred_obj in pred_objs.objects]
    gt_objs = []
    for gt_obj in gt_objects_msgs[gt_idx].data:
        # ensure it's not the ego vehicle
        if gt_obj.status.type.type == 0:
            continue
        
        gt_objs.append(DetectionObject().init_with_gt(gt_obj))
    
    # store pred_objs for frame, and gt_objs for frame, and the ego vehicle's position
    frames.append((pred_objs, gt_objs, car_pos))


# get metrics
metrics = calculate_metrics(frames, detection_range, iou_threshold=0.2)

# print the metrics
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1: {metrics['f1']}")
print(f"AP: {metrics['ap']}")