from geometry_msgs.msg import Point, Vector3
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from shapely.affinity import rotate
from benchmark_tools.math_utils import euler_from_quaternion
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import animation
from tf2_geometry_msgs import do_transform_pose
matplotlib.use('Agg')


def get_2d_polygon(bbox):
    # Compute the 4 corners of the bounding box in 2D (ignoring z)
    dx, dy = bbox.dimensions.x / 2, bbox.dimensions.y / 2
    corners = np.array([
        [-dx, -dy],
        [-dx, dy],
        [dx, dy],
        [dx, -dy]
    ])
    
    # Apply rotation
    yaw_rotation = R.from_euler('z', bbox.yaw).as_matrix()[:2, :2]  # Extract 2D rotation
    rotated_corners = np.dot(corners, yaw_rotation.T)
    
    # Translate to the bbox center
    translated_corners = rotated_corners + np.array([bbox.center.x, bbox.center.y])
    
    return Polygon(translated_corners)

def calculate_iou_2d(poly1, poly2):
    # Compute the intersection and union areas
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou

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

class BBox:
    def __init__(self, center, dimensions, quaternion):
        self.center = center
        self.dimensions = dimensions
        self.quaternion = quaternion

        # calculate the euler angles from the quaternion
        euler = euler_from_quaternion(quaternion.x, quaternion.y, quaternion.z, quaternion.w)
        self.roll = euler[0]
        self.pitch = euler[1]
        self.yaw = euler[2]
        
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
        self.bbox = self.create_bbox(gt_obj.status.bounding_box, self.pose)
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
        if pred_obj.shape.type == 0:
            self.bbox = self.create_bbox(pred_obj.shape, self.pose)
        elif pred_obj.shape.type == 2:
            self.bbox = self.create_polygon(pred_obj.shape, self.pose)
        else:
            raise ValueError("Unsupported shape type")
        self.pred = True
        return self
    
    @staticmethod
    def create_polygon(shape, pose):
        # return the polygon as shapely polygon, with the center at the pose
        points = []
        for p in shape.footprint.points:
            x = p.x + pose.position.x
            y = p.y + pose.position.y
            points.append((x, y))
        poly = Polygon(points)
        
        # apply the rotation
        yaw = euler_from_quaternion(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)[2]
        poly = rotate(poly, yaw, origin=(pose.position.x, pose.position.y))
        
        # rotate by 90 degrees to match the orientation of the bounding box
        poly = rotate(poly, 90, origin=(pose.position.x, pose.position.y))
        
        return poly
    
    @staticmethod
    def create_bbox(bbox, pose):
        temp_box = BBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=bbox.dimensions.z / 2),
            dimensions=Vector3(x=bbox.dimensions.x, y=bbox.dimensions.y, z=bbox.dimensions.z),
            quaternion=pose.orientation
        )
        return get_2d_polygon(temp_box)

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
            iou = calculate_iou_2d(pred_obj.bbox, gt_obj.bbox)
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
            dist = np.linalg.norm([pred_obj.bbox.centroid.x - gt_obj.bbox.centroid.x, pred_obj.bbox.centroid.y - gt_obj.bbox.centroid.y])
            if dist < best_dist:
                best_dist = dist
                best_match = gt_obj
        if best_match and best_dist < 10:
            pred_obj.gt_class = best_match.class_id
            
    return matches, unmatched_gt, unmatched_pred

def calculate_metrics(frames, gt_objs, det_range, ignore_objects=[]):
    # calculate metrics for iou thresholds from 0-1 in 0.1 increments
    metrics = {}
    for iou_threshold in np.arange(0, 1.1, 0.1):
        metrics[iou_threshold] = calculate_metrics_for_iou(frames, gt_objs, det_range, iou_threshold, ignore_objects)
    
    # convert the metrics to a dataframe
    df = pd.DataFrame()
    for iou_threshold, m in metrics.items():
        for class_name, metrics in m.items():
            df = pd.concat([df, pd.DataFrame({
                "class": class_name,
                "iou_threshold": iou_threshold,
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
    for pred_objs, ego_pos in frames:
        # match the objects
        matches, unmatched_gt, unmatched_pred = match_objects(pred_objs, gt_objs)
        
        # first address the unmatched ground truth objects
        # if the object is within the detection range, it is a false negative
        # otherwise, it is a true negative and should be ignored
        for gt_obj in unmatched_gt:
            if np.linalg.norm([gt_obj.bbox.centroid.x - ego_pos.x, gt_obj.bbox.centroid.y - ego_pos.y]) < det_range:
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
            # otherwise, it is a false positive
            else:
                confusion_mats[gt_obj.class_name]["FP"] += 1
    return confusion_mats


def metrics_summary(metrics_df):
    # print the metrics
    print(metrics_df.groupby(["class", "iou_threshold"]).sum())

def create_animation(frames, gt_objs, path):
    # given the frames and the ground truth objects, create an animation of the frames
    # just plot the bounding boxes of each object in each frame
    def plot_polygon(ax, polygon, edgecolor='blue', linestyle='-', label=None):
        """Helper function to plot a shapely Polygon."""
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=edgecolor, linestyle=linestyle, label=label)

    def animate(i, frames, gt_objs, ax):
        """Update function for FuncAnimation."""
        ax.clear()
        ax.set_title(f"Frame {i + 1}")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True)
        
        # Draw the ground truth bounding boxes (stationary, constant for all frames)
        for gt in gt_objs:
            if gt.obj_name == "ego":
                continue
            gt_polygon = gt.bbox
            plot_polygon(ax, gt_polygon, edgecolor='green', linestyle='--', label="Ground Truth")
        
        # Draw the detected bounding boxes for the current frame
        detections, ego_pos = frames[i]
        for detection in detections:
            detection_polygon = detection.bbox
            plot_polygon(ax, detection_polygon, edgecolor='blue', linestyle='-', label="Detection")
        
        # Draw a circle at the ego position
        ego_circle = plt.Circle((ego_pos.x, ego_pos.y), 1, color='red', fill=False, label="Ego Position")
        ax.add_artist(ego_circle)
        
        # Add legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize the animation
    ani = FuncAnimation(
        fig,
        animate,
        frames=len(frames),
        fargs=(frames, gt_objs, ax),
        interval=500,  # Interval between frames in milliseconds
        repeat=True
    )

    writervideo = animation.FFMpegWriter(fps=2)
    ani.save(path, writer=writervideo)
    plt.close()