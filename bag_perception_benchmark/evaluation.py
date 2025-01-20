from autoware_perception_msgs.msg import PredictedObjects, PredictedObject, PredictedObjectKinematics, ObjectClassification, Shape
from traffic_simulator_msgs.msg import EntityStatusWithTrajectoryArray, EntityStatus, BoundingBox
from geometry_msgs.msg import Point, Vector3
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from pydoc import locate
import argparse
import numpy as np
from benchmark_tools.eval_utils import DetectionObject, calculate_metrics, metrics_summary
import pandas as pd
import os

# parse the arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--bag_file", type=str, required=True)
argparser.add_argument("--run_id", type=str, required=True)
argparser.add_argument("--output_df", type=str, required=True)
args = argparser.parse_args()
bag_file = args.bag_file
run_id = args.run_id
output_df_path = args.output_df

# topic to read from for the predicted objects
predicted_objs_topic = "/perception/object_recognition/objects"

# topic to read from for the ground truth objects
gt_topic = "/simulation/entity/status"

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
gt_objects_msg = None
car_positions = []
objects_seen = set()
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == gt_topic:
        gt_objects_msg = msg
    
    elif topic == "/tf":
        # save the current position of the base link for later use
        for transform in msg.transforms:
            if transform.header.frame_id == "map" and transform.child_frame_id == "base_link":
                car_positions.append(transform.transform.translation)
    
    elif topic == predicted_objs_topic:
        for obj in msg.objects:
            uuid_str = "".join([f"{b:02x}" for b in obj.object_id.uuid])
            if uuid_str not in objects_seen:
                objects_seen.add(uuid_str)
        
        # append the message to the list
        # also keep track of the index of the car's last position for matching
        predicted_objects_msgs.append((msg, car_positions[-1]))

# range to consider a ground truth object as able to be detected
# if within this range and not detected, it is a false negative
detection_range = 20

# convert the messages to DetectionObject objects and create frames for the predicted objects
gt_objects = set()
frames = []
for pred_objs, car_pos in predicted_objects_msgs:
    pred_objs = [DetectionObject().init_with_pred(pred_obj) for pred_obj in pred_objs.objects]
    
    # store pred_objs for frame, and gt_objs for frame, and the ego vehicle's position
    frames.append((pred_objs, car_pos))

# convert the ground truth objects to DetectionObject objects
gt_objects = [DetectionObject().init_with_gt(gt_obj) for gt_obj in gt_objects_msg.data]

# get metrics
metrics = calculate_metrics(frames, gt_objects, detection_range, iou_threshold=0.2)
metrics_summary(metrics)

# record the metrics in a pandas dataframe
# if the file does not exist, create it
# otherwise, append to it
if not os.path.exists(output_df_path):
    df = pd.DataFrame(columns=["run_id", "obj_name", "precision", "recall", "f1", "ap"])
else:
    df = pd.read_csv(output_df_path)

# add the metrics to the dataframe
for gt_obj, metric in metrics.items():
    record = {'run_id': run_id, 'obj_name': gt_obj.obj_name, 'precision': metric["precision"], 'recall': metric["recall"], 'f1': metric["f1"], 'ap': metric["ap"]}
    df = pd.concat([df, pd.DataFrame.from_records([record])])

# write the dataframe to the output file
df.to_csv(output_df_path, index=False)
