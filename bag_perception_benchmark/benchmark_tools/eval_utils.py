import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tf2_geometry_msgs import do_transform_pose, TransformStamped
from geometry_msgs.msg import Vector3, Quaternion
from quaternion import quaternion, as_rotation_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from benchmark_tools.iou import IoU
from benchmark_tools.box import Box
from tqdm import tqdm
# matplotlib.use('Agg')


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

def create_bounding_box_for_box(dimensions, pos, orientation):
    position = np.array([pos.x, pos.y, pos.z])
    rot_mat = as_rotation_matrix(quaternion(orientation.w, orientation.x, orientation.y, orientation.z))
    scale = np.array([dimensions.x, dimensions.y, dimensions.z])
    
    return Box().from_transformation(rot_mat, position, scale)

def create_bounding_box_for_polygon(footprint, height, pos, orientation):
    position = np.array([pos.x, pos.y, pos.z])
    rot_mat = as_rotation_matrix(quaternion(orientation.w, orientation.x, orientation.y, orientation.z))
    
    # calculate bounds of the footprint for x and y dims
    footprint = np.array([[point.x, point.y, point.z] for point in footprint.points])
    min_x, max_x = np.min(footprint[:, 0]), np.max(footprint[:, 0])
    min_y, max_y = np.min(footprint[:, 1]), np.max(footprint[:, 1])
    
    scale = np.array([max_x-min_x, max_y-min_y, height])
    
    return Box().from_transformation(rot_mat, position, scale)

def calc_3d_iou(box1, box2):
    iou = IoU(box1, box2)
    return iou.iou()

class DetectionObject:
    def __init__(self):
        self.class_id = None
        self.gt_class = None
        self.class_name = None
        self.class_conf = None
        self.exist_conf = None
        self.bbox = None
        self.pose = None
        self.gt = False
        self.pred = False
        self.obj_name = None
    
    def __str__(self):
        return f"{self.class_name} {self.class_conf:.2f} {self.exist_conf:.2f} {self.bbox.center.x:.2f} {self.bbox.center.y:.2f} {self.bbox.dimensions.x:.2f} {self.bbox.dimensions.y:.2f}"
    
    def init_with_gt(self, gt_obj):
        self.class_id = entitytype_to_class[gt_obj.status.type.type]
        self.gt_class = self.class_id
        self.class_name = classes[self.class_id]
        self.class_conf = 1.0
        self.exist_conf = 1.0
        self.pose = gt_obj.status.pose
        
        pos = self.pose.position
        bbox = gt_obj.status.bounding_box
        pos.x += bbox.center.x / 2
        pos.y += bbox.center.y / 2
        pos.z += bbox.center.z / 1.2
        self.pos = np.array([pos.x, pos.y, pos.z])
        self.bbox = create_bounding_box_for_box(bbox.dimensions, pos, self.pose.orientation)
        
        self.gt = True
        self.obj_name = gt_obj.name
        return self
    
    def init_with_pred(self, pred_obj, base_link_tf):
        # get highest confidence class
        cls_w_conf = [(c.label, c.probability) for c in pred_obj.classification]
        cls_w_conf.sort(key=lambda x: x[1], reverse=True)
        self.class_id = cls_w_conf[0][0]
        self.gt_class = None # will be set when matching
        self.class_name = classes[self.class_id]
        self.class_conf = cls_w_conf[0][1]
        self.exist_conf = pred_obj.existence_probability
        self.pose = do_transform_pose(pred_obj.kinematics.pose_with_covariance.pose, base_link_tf)
        self.pos = np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z])
        if pred_obj.shape.type == 0:
            self.bbox = create_bounding_box_for_box(pred_obj.shape.dimensions, self.pose.position, self.pose.orientation)
        elif pred_obj.shape.type == 2:
            self.bbox = create_bounding_box_for_polygon(pred_obj.shape.footprint, pred_obj.shape.dimensions.z, self.pose.position, self.pose.orientation)
        else:
            raise ValueError("Unsupported shape type")
        self.pred = True
        return self

def match_objects(pred_objs, gt_objs):
    # for each pair of pred_obj and gt_obj, calculate the IoU and store the best match
    # ensure that each gt_obj is only matched once
    matches = []
    unmatched_gt = gt_objs.copy()
    unmatched_pred = pred_objs.copy()
    for pred_obj in pred_objs:
        best_iou = 0
        best_match = None
        for gt_obj in unmatched_gt:
            iou = calc_3d_iou(pred_obj.bbox, gt_obj.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = gt_obj
        if best_match:
            matches.append((pred_obj, best_match, best_iou))
            unmatched_gt.remove(best_match)
            unmatched_pred.remove(pred_obj)
    
    # set the gt_class for each pred_obj to the nearest gt_obj
    for pred_obj in unmatched_pred:
        # find the nearest gt_obj
        best_dist = np.inf
        best_match = None
        for gt_obj in unmatched_gt:
            dist = np.linalg.norm(pred_obj.pos - gt_obj.pos)
            if dist < best_dist:
                best_dist = dist
                best_match = gt_obj
        if best_match and best_dist < 10:
            pred_obj.gt_class = best_match.class_id
            
    return matches, unmatched_gt, unmatched_pred

def calculate_metrics(frames, gt_objs, det_range, ignore_objects=[]):
    # calculate metrics for iou thresholds from 0-1 in 0.1 increments
    metrics = {}
    for iou_threshold in tqdm(np.arange(0, 1.1, 0.1)):
        metrics[iou_threshold] = calculate_metrics_for_iou(frames, gt_objs, det_range, iou_threshold, ignore_objects)
    
    # convert the metrics to a dataframe
    df = pd.DataFrame()
    for iou_threshold, m in metrics.items():
        for class_name, metrics in m.items():
            df = pd.concat([df, pd.DataFrame({
                "class": class_name,
                "iou_threshold": iou_threshold,
                "avg_conf": metrics["avg_conf"],
                "tp": [metrics["TP"]],
                "fp": [metrics["FP"]],
                "fn": [metrics["FN"]]
            }, index=[0])])
    
    return df

def calculate_metrics_for_iou(frames, gt_objs, det_range, iou_threshold=0.2, ignore_objects=[]):
    # Calculate the tp, fp, and fn for each class at the given IoU threshold
    
    # remove ego from the ground truth objects
    gt_objs = [gt_obj for gt_obj in gt_objs if gt_obj.class_id != 0]
    
    # create a confusion matrix for each class
    gt_classes = set([gt_obj.class_name for gt_obj in gt_objs])
    confusion_mats = {class_name: {"TP": 0, "FP": 0, "FN": 0} for class_name in gt_classes if class_name}
    confidences = {class_name: [] for class_name in gt_classes if class_name}
    for pred_objs, ego_pos in frames:
        # match the objects
        matches, unmatched_gt, unmatched_pred = match_objects(pred_objs, gt_objs)
        
        # first address the unmatched ground truth objects
        # if the object is within the detection range, it is a false negative
        # otherwise, it is a true negative and should be ignored
        for gt_obj in unmatched_gt:
            if np.linalg.norm([gt_obj.pos[0] - ego_pos.x, gt_obj.pos[1] - ego_pos.y]) < det_range:
                if gt_obj.class_name not in ignore_objects: # ignore objects that are not in the classes to be evaluated
                    confusion_mats[gt_obj.class_name]["FN"] += 1
        
        # address the unmatched predicted objects 
        # count as false positives for the class that they are closest to (gt_class from matching)
        for pred_obj in unmatched_pred:
            if pred_obj.gt_class:
                confusion_mats[classes[pred_obj.gt_class]]["FP"] += 1
        
        # now address the matched objects
        for pred_obj, gt_obj, iou in matches:
            if gt_obj.class_name in ignore_objects: # ignore objects that are not in the classes to be evaluated
                continue
            # if the IoU is above the threshold, it is a true positive
            if iou > iou_threshold:
                confusion_mats[gt_obj.class_name]["TP"] += 1
                confidences[gt_obj.class_name].append(pred_obj.exist_conf)
            # otherwise, it is a false positive
            else:
                confusion_mats[gt_obj.class_name]["FP"] += 1
    
    # add the average confidence for each class
    for class_name, confs in confidences.items():
        if len(confs) > 0:
            confusion_mats[class_name]["avg_conf"] = np.mean(confs)
        else:
            confusion_mats[class_name]["avg_conf"] = 0
    
    return confusion_mats

def metrics_summary(metrics_df):
    # print the metrics
    print(metrics_df.groupby(["class", "iou_threshold"]).sum())

def plot_bounding_box(bbox, ax=None, color='b', label=None):
    """
    Plots a 3D bounding box using matplotlib.

    Args:
        bbox: A numpy array of shape (8, 3) representing the vertices of the bounding box.
        ax: A matplotlib 3D axes object. If None, a new figure and axes will be created.
        color: The color of the bounding box edges.
        label: A label for the bounding box (for legend).

    Returns:
        The matplotlib 3D axes object.
    """
    bbox = bbox.vertices
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Define edges of the bounding box (indices of the vertices)
    edges = [
        [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
        [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
        [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    ]

    # Plot the edges
    for edge in edges:
        ax.plot(bbox[edge, 0], bbox[edge, 1], bbox[edge, 2], color=color)

    # If a label is provided, add it to the plot (for legend)
    if label:
        # Get the center of the bounding box for the label position
        center = np.mean(bbox, axis=0)
        ax.text(center[0], center[1], center[2], label, color=color)

    return ax
