from benchmark_tools.math_utils import euler_from_quaternion, calculate_iou_2d, get_2d_polygon
from geometry_msgs.msg import Point, Vector3
import numpy as np
from sklearn.metrics import precision_recall_curve, auc

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
        self.class_name = None
        self.class_conf = None
        self.exist_conf = None
        self.bbox = None
        self.pose = None
        self.gt = False
        self.pred = False
        self.obj_id = None
    
    def __str__(self):
        return f"{self.class_name} {self.class_conf:.2f} {self.exist_conf:.2f} {self.bbox.center.x:.2f} {self.bbox.center.y:.2f} {self.bbox.dimensions.x:.2f} {self.bbox.dimensions.y:.2f}"
    
    def init_with_gt(self, gt_obj):
        self.class_id = entitytype_to_class[gt_obj.status.type.type]
        self.class_name = classes[self.class_id]
        self.class_conf = 1.0
        self.exist_conf = 1.0
        self.pose = gt_obj.status.pose
        self.bbox = self.gt_bbox(gt_obj.status.bounding_box, self.pose)
        self.gt = True
        self.obj_name = gt_obj.name
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
        return BBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=dimensions.z / 2),
            dimensions=Vector3(x=max_x - min_x, y=max_y - min_y, z=dimensions.z),
            quaternion=pose.orientation
        )
    
    @staticmethod
    def pred_bbox(bbox, pose):
        # put center at the pose and swap x and y
        return BBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=bbox.dimensions.z / 2),
            dimensions=Vector3(x=bbox.dimensions.x, y=bbox.dimensions.y, z=bbox.dimensions.z),
            quaternion=pose.orientation
        )
    
    @staticmethod
    def gt_bbox(bbox, pose):
        # put center at the pose and swap x and y
        return BBox(
            center=Point(x=pose.position.x, y=pose.position.y, z=bbox.dimensions.z / 2),
            dimensions=Vector3(x=bbox.dimensions.x, y=bbox.dimensions.y, z=bbox.dimensions.z),
            quaternion=pose.orientation
        )

def match_objects(pred_objs, gt_objs):
    # for each pair of pred_obj and gt_obj, calculate the IoU and store the best match
    # ensure that each gt_obj is only matched once
    matches = []
    unmatched_gt = gt_objs.copy()
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
            
    return matches, unmatched_gt
    

def calculate_metrics(frames, gt_objs, det_range, iou_threshold=0.2):
    # Calculate the recall, precision, f1, and AP for each object in the ground truth
    
    # remove ego from the ground truth objects
    gt_objs = [gt_obj for gt_obj in gt_objs if gt_obj.class_id != 0]
    
    # create a confusion matrix for each ground truth object
    confusion_mats = {gt_obj: {"TP": 0, "FP": 0, "FN": 0} for gt_obj in gt_objs}
    for pred_objs, ego_pos in frames:
        # match the objects
        matches, unmatched_gt = match_objects(pred_objs, gt_objs)
        
        # first address the unmatched ground truth objects
        # if the object is within the detection range, it is a false negative
        # otherwise, it is a true negative and should be ignored
        for gt_obj in unmatched_gt:
            if np.linalg.norm([gt_obj.bbox.center.x - ego_pos.x, gt_obj.bbox.center.y - ego_pos.y]) < det_range:
                confusion_mats[gt_obj]["FN"] += 1
        
        # now address the matched objects
        for pred_obj, gt_obj, iou in matches:
            if iou > iou_threshold:
                confusion_mats[gt_obj]["TP"] += 1
            else:
                confusion_mats[gt_obj]["FP"] += 1
        
    # calculate the precision, recall, and f1 for each object
    metrics = {x: {"precision": 0, "recall": 0, "f1": 0, "pr_auc": 0} for x in gt_objs}
    for gt_obj in gt_objs:
        TP = confusion_mats[gt_obj]["TP"]
        FP = confusion_mats[gt_obj]["FP"]
        FN = confusion_mats[gt_obj]["FN"]
        if TP + FP > 0:
            metrics[gt_obj]["precision"] = TP / (TP + FP)
        if TP + FN > 0:
            metrics[gt_obj]["recall"] = TP / (TP + FN)
        if metrics[gt_obj]["precision"] + metrics[gt_obj]["recall"] > 0:
            metrics[gt_obj]["f1"] = 2 * (metrics[gt_obj]["precision"] * metrics[gt_obj]["recall"]) / (metrics[gt_obj]["precision"] + metrics[gt_obj]["recall"])
    
    # calculate AP for each object
    pr_curves = calculate_pr_curve(frames, gt_objs, det_range, iou_threshold)
    for gt_obj in gt_objs:
        precision = pr_curves[gt_obj]["precision"]
        recall = pr_curves[gt_obj]["recall"]
        metrics[gt_obj]["ap"] = auc(recall, precision)
    
    return metrics

def calculate_pr_curve(frames, gt_objs, det_range, iou_threshold=0.2):
    # Calculate the precision recall curve for each ground truth object so that we can calculate the AUC for each object
    pr_curves = {gt_obj: {"precision": [], "recall": []} for gt_obj in gt_objs}
    
    for pred_objs, ego_pos in frames:
        # match the objects
        matches, unmatched_gt = match_objects(pred_objs, gt_objs)
        
        # first address the unmatched ground truth objects
        # if the object is within the detection range, it is a false negative
        # otherwise, it is a true negative and should be ignored
        for gt_obj in unmatched_gt:
            if np.linalg.norm([gt_obj.bbox.center.x - ego_pos.x, gt_obj.bbox.center.y - ego_pos.y]) < det_range:
                pr_curves[gt_obj]["recall"].append(0)
                pr_curves[gt_obj]["precision"].append(0)
        
        # now address the matched objects
        for pred_obj, gt_obj, iou in matches:
            if iou > iou_threshold:
                pr_curves[gt_obj]["recall"].append(1)
                pr_curves[gt_obj]["precision"].append(1)
            else:
                pr_curves[gt_obj]["recall"].append(0)
                pr_curves[gt_obj]["precision"].append(1)
    
    # calculate the precision recall curve for each object
    for gt_obj in gt_objs:
        precision, recall, _ = precision_recall_curve(pr_curves[gt_obj]["recall"], pr_curves[gt_obj]["precision"])
        pr_curves[gt_obj]["precision"] = precision
        pr_curves[gt_obj]["recall"] = recall
    
    return pr_curves

def metrics_summary(metrics):
    for obj, m in metrics.items():
        print(f"{obj.obj_name}:")
        print(f"  Precision: {m['precision']:.2f}")
        print(f"  Recall: {m['recall']:.2f}")
        print(f"  F1: {m['f1']:.2f}")
        print(f"  AP: {m['ap']:.2f}")
        print()