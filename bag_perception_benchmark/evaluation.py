from autoware_perception_msgs.msg import PredictedObjects, PredictedObject, PredictedObjectKinematics, ObjectClassification, Shape
from traffic_simulator_msgs.msg import EntityStatusWithTrajectoryArray, EntityStatus, BoundingBox
from geometry_msgs.msg import Point, Vector3
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter
from rclpy.serialization import deserialize_message
from pydoc import locate
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import numpy as np

# REFERENCE MESSAGES
'''
PredictedObject:
unique_identifier_msgs/UUID object_id
float32 existence_probability
ObjectClassification[] classification
PredictedObjectKinematics kinematics
Shape shape

PredictedObjectKinematics:
geometry_msgs/PoseWithCovariance initial_pose_with_covariance
geometry_msgs/TwistWithCovariance initial_twist_with_covariance
geometry_msgs/AccelWithCovariance initial_acceleration_with_covariance
PredictedPath[] predicted_paths

Shape:
uint8 BOUNDING_BOX=0
uint8 CYLINDER=1
uint8 POLYGON=2
uint8 type
geometry_msgs/Polygon footprint
geometry_msgs/Vector3 dimensions

ObjectClassification:
uint8 UNKNOWN = 0
uint8 CAR = 1
uint8 TRUCK = 2
uint8 BUS = 3
uint8 TRAILER = 4
uint8 MOTORCYCLE = 5
uint8 BICYCLE = 6
uint8 PEDESTRIAN = 7
uint8 label
float32 probability

EntityStatusWithTrajectoryArray:
traffic_simulator_msgs/EntityStatusWithTrajectory[] data

EntityStatusWithTrajectory:
float64 time 0
string name
traffic_simulator_msgs/EntityStatus status
traffic_simulator_msgs/WaypointsArray waypoint
geometry_msgs/Pose[] goal_pose
bool obstacle_find false
traffic_simulator_msgs/Obstacle obstacle

EntityStatus:
traffic_simulator_msgs/EntityType type
traffic_simulator_msgs/EntitySubtype subtype
float64 time 0
string name
traffic_simulator_msgs/BoundingBox bounding_box
traffic_simulator_msgs/ActionStatus action_status
geometry_msgs/Pose pose
traffic_simulator_msgs/LaneletPose lanelet_pose
bool lanelet_pose_valid true

EntityType:
uint8 EGO = 0
uint8 VEHICLE = 1
uint8 PEDESTRIAN = 2
uint8 MISC_OBJECT = 3
uint8 type

BoundingBox:
geometry_msgs/Point center
geometry_msgs/Vector3 dimensions
'''


# topic to read from for the predicted objects
predicted_objs_topic = "/perception/object_recognition/objects"

# topic to read from for the ground truth objects
gt_topic = "/simulation/entity/status"

classes = {
    0: "UNKNOWN",
    1: "CAR",
    2: "TRUCK",
    3: "BUS",
    4: "TRAILER",
    5: "MOTORCYCLE",
    6: "BICYCLE",
    7: "PEDESTRIAN"
}

entitytype_to_class = {
    0: 0,
    1: 1, 
    2: 7,
    3: 0
}

class DetectionObject:
    def __init__(self):
        self.class_id = None
        self.class_name = None
        self.class_conf = None
        self.exist_conf = None
        self.bbox = None
        self.pose = None
        self.gt = False
        self.pred = False
    
    def init_with_gt(self, gt_obj):
        self.class_id = entitytype_to_class[gt_obj.status.type.type]
        self.class_name = classes[self.class_id]
        self.class_conf = 1.0
        self.exist_conf = 1.0
        self.pose = gt_obj.status.pose
        self.bbox = self.gt_bbox(gt_obj.status.bounding_box, self.pose)
        self.gt = True
        return self
    
    def init_with_pred(self, pred_obj):
        # get highest confidence class
        cls_w_conf = [(c.label, c.probability) for c in pred_obj.classification]
        cls_w_conf.sort(key=lambda x: x[1], reverse=True)
        self.class_id = cls_w_conf[0][0]
        self.class_name = classes[self.class_id]
        self.class_conf = cls_w_conf[0][1]
        self.exist_conf = pred_obj.existence_probability
        self.pose = pred_obj.kinematics.initial_pose_with_covariance.pose
        if pred_obj.shape.type == 0:
            self.bbox = self.pred_bbox(pred_obj.shape, self.pose)
        elif pred_obj.shape.type == 2:
            self.bbox = self.polygon_to_bbox(pred_obj.shape, self.pose)
        else:
            raise ValueError("Unsupported shape type")
        
        self.pred = True
        return self
    
    @staticmethod
    def polygon_to_bbox(shape, pose):
        footprint = shape.footprint
        dimensions = shape.dimensions
        min_x = min([p.x for p in footprint.points])
        max_x = max([p.x for p in footprint.points])
        min_y = min([p.y for p in footprint.points])
        max_y = max([p.y for p in footprint.points])
        return BoundingBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=dimensions.z / 2),
            dimensions=Vector3(x=max_y - min_y, y=max_x - min_x, z=dimensions.z)
        )
    
    @staticmethod
    def pred_bbox(bbox, pose):
        # put center at the pose and swap x and y
        return BoundingBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=bbox.dimensions.z / 2),
            dimensions=Vector3(x=bbox.dimensions.y, y=bbox.dimensions.x, z=bbox.dimensions.z)
        )
    
    @staticmethod
    def gt_bbox(bbox, pose):
        # put center at the pose and swap x and y
        return BoundingBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=bbox.dimensions.z / 2),
            dimensions=Vector3(x=bbox.dimensions.y, y=bbox.dimensions.x, z=bbox.dimensions.z)
        )

def calculate_iou(bbox1, bbox2):
    # calculate the intersection over union of two bounding boxes in 2d space
    min_x1 = bbox1.center.x - bbox1.dimensions.x / 2
    max_x1 = bbox1.center.x + bbox1.dimensions.x / 2
    min_y1 = bbox1.center.y - bbox1.dimensions.y / 2
    max_y1 = bbox1.center.y + bbox1.dimensions.y / 2
    
    min_x2 = bbox2.center.x - bbox2.dimensions.x / 2
    max_x2 = bbox2.center.x + bbox2.dimensions.x / 2
    min_y2 = bbox2.center.y - bbox2.dimensions.y / 2
    max_y2 = bbox2.center.y + bbox2.dimensions.y / 2
    
    x_overlap = max(0, min(max_x1, max_x2) - max(min_x1, min_x2))
    y_overlap = max(0, min(max_y1, max_y2) - max(min_y1, min_y2))
    
    intersection = x_overlap * y_overlap
    union = bbox1.dimensions.x * bbox1.dimensions.y + bbox2.dimensions.x * bbox2.dimensions.y - intersection
    return intersection / union

def match_objects(pred_objs, gt_objs):
    # match the predicted objects with the ground truth objects
    matches = []
    gt_taken = [False] * len(gt_objs)
    for pred_obj in pred_objs:
        max_iou = 0
        match = None
        for gt_obj in gt_objs:
            if gt_taken[gt_objs.index(gt_obj)]:
                continue
            iou = calculate_iou(pred_obj.bbox, gt_obj.bbox)
            if iou > max_iou:
                max_iou = iou
                match = gt_obj
        if pred_obj and match:
            matches.append((pred_obj, match, max_iou))
            gt_taken[gt_objs.index(match)] = True
            # print(f"Matched {pred_obj.class_name} with {match.class_name} with IoU {max_iou}")
    return matches


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
filter = StorageFilter(topics=[predicted_objs_topic, gt_topic])
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
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == gt_topic:
        gt_objects_msgs.append(msg)
    
    elif topic == predicted_objs_topic:
        # only consider messages with objects detected
        if len(msg.objects) == 0:
            continue
        predicted_objects_msgs.append((msg, len(gt_objects_msgs)-1))
        break

# after we get the first predicted objects message, read the rest of the messages
while reader.has_next():
    (topic, msg, t) = reader.read_next()
    msg = deserialize_message(msg, typestr_to_type[topic])
    if topic == gt_topic:
        gt_objects_msgs.append(msg)
    
    elif topic == predicted_objs_topic:
        # append the message to the list, and also keep track of the index of the last gt message for later matching
        predicted_objects_msgs.append((msg, len(gt_objects_msgs)-1))

# convert the messages to DetectionObject objects
gt_obj_frames = []
pred_obj_frames = []
for pred_objs, gt_idx in predicted_objects_msgs:
    pred_objs = [DetectionObject().init_with_pred(pred_obj) for pred_obj in pred_objs.objects]
    gt_objs = [DetectionObject().init_with_gt(gt_obj) for gt_obj in gt_objects_msgs[gt_idx].data if gt_obj.status.type.type != 0]
    pred_obj_frames.append(pred_objs)
    gt_obj_frames.append(gt_objs)


# # calculate 3d object detection metrics
# # 1. IoU
# # 2. Precision
# # 3. Recall
# # 4. F1 score
# # 5. Average Precision
# # 6. Average Recall

threshold = 0.01

def calculate_metrics(pred_objs, gt_objs):
    matches = match_objects(pred_objs, gt_objs)
    tp = 0
    fp = 0
    fn = 0
    total_iou = 0
    for pred_obj, gt_obj, iou in matches:
        if iou >= threshold:
            tp += 1
            total_iou += iou
        else:
            fp += 1
    fn = len(gt_objs) - tp
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    avg_precision = total_iou / tp if tp > 0 else 0
    avg_recall = total_iou / len(gt_objs) if len(gt_objs) > 0 else 0
    return precision, recall, f1, avg_precision, avg_recall

metrics = []
for pred_objs, gt_objs in zip(pred_obj_frames, gt_obj_frames):
    precision, recall, f1, avg_precision, avg_recall = calculate_metrics(pred_objs, gt_objs)
    metrics.append((precision, recall, f1, avg_precision, avg_recall))

# print the metric summary
precision = sum([m[0] for m in metrics]) / len(metrics)
recall = sum([m[1] for m in metrics]) / len(metrics)
f1 = sum([m[2] for m in metrics]) / len(metrics)
avg_precision = sum([m[3] for m in metrics]) / len(metrics)
avg_recall = sum([m[4] for m in metrics]) / len(metrics)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
print(f"Average Precision: {avg_precision}")
print(f"Average Recall: {avg_recall}")

