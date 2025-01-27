from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from pydoc import locate
import argparse
import numpy as np
from benchmark_tools.eval_utils import DetectionObject, calculate_metrics, metrics_summary, create_animation
import pandas as pd
import os

# parse the arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--bag_dir", type=str, required=True)
argparser.add_argument("--generate_animation", action="store_true")
args = argparser.parse_args()
bag_dir = args.bag_dir
generate_animation = args.generate_animation
output_df_path = os.path.join(bag_dir, "metrics.csv")

# topic to read from for the detected objects
detected_objs_topic = "/perception/object_recognition/detection/objects"

# topic to read from for the ground truth objects
gt_topic = "/simulation/entity/status"

bags = os.listdir(bag_dir)
bags = [bag for bag in bags if bag.endswith(".db3")]

for bag in bags:
    
    bag_file = os.path.join(bag_dir, bag)

    # read the bag
    reader = SequentialReader()
    storage_options = StorageOptions(uri=bag_file, storage_id="sqlite3")
    converter_options = ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    # filter to only read the topics we care about
    filter = StorageFilter(topics=[detected_objs_topic, gt_topic, "/tf"])
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
                    car_positions.append(transform)
        
        elif topic == detected_objs_topic:
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
        pred_objs = [DetectionObject().init_with_pred(pred_obj, car_pos) for pred_obj in pred_objs.objects]
        
        # store pred_objs for frame, and gt_objs for frame, and the ego vehicle's position
        frames.append((pred_objs, car_pos.transform.translation))

    # convert the ground truth objects to DetectionObject objects
    gt_objects = [DetectionObject().init_with_gt(gt_obj) for gt_obj in gt_objects_msg.data]

    # objects to ignore when calculating metrics
    ignore_objects = ["Pedestrian1", "Van0"]

    # get metrics for the frames
    results_df = calculate_metrics(frames, gt_objects, detection_range)

    # print the metrics
    metrics_summary(results_df)

    # record the metrics in a pandas dataframe
    # if the file does not exist, create it
    # otherwise, append to it
    if not os.path.exists(output_df_path):
        df = pd.DataFrame(columns=["run_id"] + list(results_df.columns))
    else:
        df = pd.read_csv(output_df_path)

    run_id = os.path.basename(bag_file).split(".")[0].replace("_output_0", "")

    # add the run_id column to the results dataframe
    results_df["run_id"] = run_id

    # concat the new metrics to the dataframe
    df = pd.concat([df, results_df])

    # write the dataframe to the output file
    df.to_csv(output_df_path, index=False)

    print(f"Metrics written to {output_df_path} for run {run_id}")
    
    # generate animation if specified
    if generate_animation:
        print(f"Generating animation for run {run_id}")
        animation_path = os.path.join(bag_dir, f"{run_id}.mp4")
        create_animation(frames, gt_objects, animation_path)
