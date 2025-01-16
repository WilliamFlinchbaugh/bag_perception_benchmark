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
    # match the predicted objects with the ground truth objects
    # also return the unmatched ground truth objects
    
    pred_matches = {pred_obj: None for pred_obj in pred_objs}
    gt_taken = {gt_obj: False for gt_obj in gt_objs}
    
    # for each pair of pred_obj and gt_obj, calculate the IoU
    for pred_obj in pred_objs:
        for gt_obj in gt_objs:
            iou = calculate_iou_2d(pred_obj.bbox, gt_obj.bbox)
            if pred_matches[pred_obj] is None or iou > pred_matches[pred_obj][1] and not gt_taken[gt_obj]:
                pred_matches[pred_obj] = (gt_obj, iou)
                gt_taken[gt_obj] = True
    
    # return the matches and the unmatched ground truth objects
    matches = [(pred_obj, gt_obj, iou) for pred_obj, (gt_obj, iou) in pred_matches.items() if gt_obj is not None]
    unmatched_gt = [gt_obj for gt_obj, taken in gt_taken.items() if not taken]
    
    return matches, unmatched_gt
    

def get_confusion_matrix(pred_objs, gt_objs, ego_pos, det_range, iou_threshold=0.2):
    """Find the confusion matrix for a set of predicted objects and ground truth objects

    Args:
        pred_objs (List[DetectionObj]): List of predicted objects
        gt_objs (List[DetectionObj]): List of ground truth objects
        ego_pos (Point): Ego vehicle position
        det_range (float): Detection range, objects within this range are considered detectable and if not detected are false negatives
        iou_threshold (float, optional): IoU threshold for a match. Defaults to 0.2.

    Returns:
        Tuple[List[DetectionObj], List[DetectionObj], List[DetectionObj], float]: Tuple of true positives, false positives, false negatives, and average IoU
    """
    matches, unmatched_gt = match_objects(pred_objs, gt_objs)
    
    # calculate tp, fp, fn
    tp = []
    fp = []
    fn = []
    
    # check if each match is a tp or fp
    for pred_obj, gt_obj, iou in matches:
        if iou > iou_threshold:
            tp.append(pred_obj)
        else:
            fp.append(pred_obj)

    # check if each unmatched gt object is a fn (if within detection range)
    for gt_obj in unmatched_gt:
        if np.linalg.norm([ego_pos.x - gt_obj.bbox.center.x, ego_pos.y - gt_obj.bbox.center.y]) < det_range:
            fn.append(gt_obj)
    
    avg_iou = sum([iou for _, _, iou in matches]) / len(matches) if len(matches) > 0 else 0

    return tp, fp, fn, avg_iou


def calculate_metrics(frames, det_range, iou_threshold=0.2):
    # calculate the precision, recall, f1, and ap for a set of frames
    tp_list = []
    fp_list = []
    fn_list = []
    avg_ious = []
    conf_scores = []
    true_labels = []

    for frame in frames:
        pred_objs, gt_objs, ego_pos = frame

        tp, fp, fn, avg_iou = get_confusion_matrix(pred_objs, gt_objs, ego_pos, det_range, iou_threshold)
        tp_list.extend(tp)
        fp_list.extend(fp)
        fn_list.extend(fn)
        avg_ious.append(avg_iou)

        for obj in tp:
            conf_scores.append(obj.exist_conf)
            true_labels.append(1)
        for obj in fp:
            conf_scores.append(obj.exist_conf)
            true_labels.append(0)
        for obj in fn:
            conf_scores.append(0)
            true_labels.append(1)

    precision = len(tp_list) / (len(tp_list) + len(fp_list)) if (len(tp_list) + len(fp_list)) > 0 else 0
    recall = len(tp_list) / (len(tp_list) + len(fn_list)) if (len(tp_list) + len(fn_list)) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    precision_curve, recall_curve, _ = precision_recall_curve(true_labels, conf_scores)
    ap = auc(recall_curve, precision_curve)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'ap': ap,
        'avg_iou': sum(avg_ious) / len(avg_ious) if len(avg_ious) > 0 else 0
    }