#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from openvino import Core
import time 

class YoloV5OpenVINONode(Node):
    def __init__(self):
        super().__init__('yolov5_node')

        # Parameters
        self.model_path = self.declare_parameter('model_path', '/home/cuon/yolov5_ros2_ws/src/yolov5_ros/models/yolov5s.xml').value
        self.class_names_path = self.declare_parameter('class_names_path', '/home/cuon/yolov5_ros2_ws/src/yolov5_ros/yolov5/data/coco128.yaml').value
        self.camera_topic = self.declare_parameter('camera_topic', '/camera/image_raw').value
        self.conf_threshold = self.declare_parameter('confidence_threshold', 0.25).value
        self.iou_threshold = self.declare_parameter('iou_threshold', 0.45).value

        # self.start_time = time.time()

        # Calibration camera parameters
        self.cameraMatrix = np.array([
            [797.4981, 0, 323.1862],
            [0, 797.8648, 263.6120],
            [0, 0, 1]
        ], dtype=np.float64)

        self.rvec = np.array([
            [2.215027358148228],
            [0.01428595002737916],
            [-0.03574176312047295]
        ])

        self.tvec = np.array([
            [-2.614781637432051],
            [142.2897382562742],
            [681.1037771565396]
        ])

        # Axis(origin, X, Y, Z)
        self.axis = np.array([
            (-250, 0, 0),     # Gốc tọa độ
            (-150, 0, 0),     # Trục X
            (-250, 200, 0),   # Trục Y
            (-250, 0, 50)     # Trục Z
        ], dtype=np.float32)

        # New axis with Z = 61.2
        self.axis_new_plane = np.array([
            (0, 0, 61.2),         # Gốc tọa độ mới
            (200, 0, 61.2),       # Trục X1
            (0, 500, 61.2),       # Trục Y1
            (0, 0, 100)         # Trục Z1
        ], dtype=np.float32)

        self.axis_labels = ['X', 'Y', 'Z']
        self.new_axis_labels = ["X1", "Y1", "Z1"]
        self.axis_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR
        self.new_axis_colors = [(255, 0, 255), (0, 255, 255), (255, 255, 0)]  # BGR

        # Load labels
        with open(self.class_names_path, 'r') as f:
            self.class_names = yaml.safe_load(f)['names']

        # OpenVINO model
        core = Core()
        model = core.read_model(self.model_path)
        config = {"PERFORMANCE_HINT": "THROUGHPUT"}
        self.compiled_model = core.compile_model(model, device_name="GPU", config= config)
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        self.model_input_size = self.input_layer.shape[2]  # (1, 3, 640, 640)

        # ROS
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image, self.camera_topic, self.image_callback, 10
        )

        # For FPS and smoothing
        self.prev_time = time.time()
        self.target_centers = []
        self.MAX_VALUES = 20

    # def non_max_suppression_openvino(self, boxes, scores, iou_threshold=0.45):
    #     if len(boxes) == 0:
    #         return []
    #     boxes = boxes.astype(np.float32)
    #     x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    #     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #     order = scores.argsort()[::-1]
    #     keep = []
    #     while order.size > 0:
    #         i = order[0]
    #         keep.append(i)
    #         xx1 = np.maximum(x1[i], x1[order[1:]])
    #         yy1 = np.maximum(y1[i], y1[order[1:]])
    #         xx2 = np.minimum(x2[i], x2[order[1:]])
    #         yy2 = np.minimum(y2[i], y2[order[1:]])
    #         w = np.maximum(0.0, xx2 - xx1 + 1)
    #         h = np.maximum(0.0, yy2 - yy1 + 1)
    #         inter = w * h
    #         iou = inter / (areas[i] + areas[order[1:]] - inter)
    #         inds = np.where(iou <= iou_threshold)[0]
    #         order = order[inds + 1]
    #     return keep

    def calculate_x_from_y(self, y, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        x = (y - b) / a
        return x

    def plot_boxes(self, frame, boxes, labels, scores):
        max_y = -1
        target_center = None
        filtered_target_center = None
        len_box = len(boxes)

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            if y2 > max_y:
                max_y = y2
                target_center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Draw Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Show class name and conf
            cv2.putText(frame, f"{self.class_names[labels[i]]} {scores[i]:.2f}",
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Calculate the center of filtered object
        if target_center:
            cv2.circle(frame, target_center, 5, (255, 0, 0), -1)
            self.target_centers.append(target_center)

            if len(self.target_centers) > self.MAX_VALUES:
                self.target_centers.pop(0)

            if len(self.target_centers) == self.MAX_VALUES:
                avg_x = sum([c[0] for c in self.target_centers]) // self.MAX_VALUES
                avg_y = sum([c[1] for c in self.target_centers]) // self.MAX_VALUES
                filtered_target_center = (avg_x, avg_y)
            else:
                filtered_target_center = target_center

            # Object's information
            X1 = int(self.calculate_x_from_y(filtered_target_center[1], (318, 379), (331, 46)))
            cv2.circle(frame, (X1, filtered_target_center[1]), 5, (0, 255, 255), -1)
            cv2.putText(frame, f"X1{str((X1, filtered_target_center[1]))}",
                        (X1 - 150, filtered_target_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.circle(frame, filtered_target_center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"X{str(filtered_target_center)}",
                        (filtered_target_center[0] + 10, filtered_target_center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            cv2.line(frame, filtered_target_center, (320, 379), (0, 150, 255), 2)
            cv2.line(frame, filtered_target_center, (X1, filtered_target_center[1]), (150, 0, 255), 2)
            
        return frame, (filtered_target_center[0], filtered_target_center[1], len_box) if filtered_target_center else None

    # Draw axes function
    def draw_axes(self, frame, axis, axis_labels, color_set):
        imgpts, _ = cv2.projectPoints(axis, self.rvec, self.tvec, self.cameraMatrix, None)
        origin_img = tuple(imgpts[0].ravel().astype(int))

        for i, color in enumerate(color_set):
            axis_img = tuple(imgpts[i + 1].ravel().astype(int))
            cv2.arrowedLine(frame, origin_img, axis_img, color, 2, tipLength=0.1)
            cv2.putText(frame, axis_labels[i], axis_img, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame
       
    def image_callback(self, msg):
        now = time.time()
        fps = 1.0 / (now - self.prev_time)
        self.prev_time = now

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        original_h, original_w = frame.shape[:2]

        # Preprocess
        image = cv2.resize(frame, (self.model_input_size, self.model_input_size), interpolation=cv2.INTER_AREA)
        input_tensor = np.moveaxis(image, -1, 0)[np.newaxis, ...] / 255.0
        input_tensor = input_tensor.astype(np.float32, copy=False)

        # Inference
        result = self.compiled_model([input_tensor])[self.output_layer]
        output = result.squeeze()
        if output.ndim == 3:
            output = output[0]

        boxes, labels, scores = [], [], []
        for pred in output:
            conf = pred[4]
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            score = class_scores[class_id] * conf
            if score > self.conf_threshold:
                x_center, y_center, w, h = pred[:4]
                x1 = int((x_center - w / 2) * original_w / self.model_input_size)
                y1 = int((y_center - h / 2) * original_h / self.model_input_size)
                x2 = int((x_center + w / 2) * original_w / self.model_input_size)
                y2 = int((y_center + h / 2) * original_h / self.model_input_size)
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
                scores.append(float(score))

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)

        # boxes_np = np.array(boxes)
        # scores_np = np.array(scores)
        # indices = self.non_max_suppression_openvino(boxes_np, scores_np, self.iou_threshold)

        if isinstance(indices, np.ndarray):
            indices = indices.flatten().tolist()
        elif isinstance(indices, list) and isinstance(indices[0], list):
            indices = [i[0] for i in indices]
        elif isinstance(indices, list):
            indices = indices
        else:
            indices = []

        filtered_boxes = [boxes[i] for i in indices] if indices else []
        filtered_labels = [labels[i] for i in indices] if indices else []
        filtered_scores = [scores[i] for i in indices] if indices else []

        frame, center_info = self.plot_boxes(frame, filtered_boxes, filtered_labels, filtered_scores)

            # Publish ra topic ROS
            # msg_out = Int32MultiArray()
            # msg_out.data = [filtered_target_center[0], filtered_target_center[1], len(filtered_boxes)]
            # self.target_pub.publish(msg_out)

        # Vẽ FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Draw axes
        frame = self.draw_axes(frame, self.axis, self.axis_labels, self.axis_colors)

        frame = self.draw_axes(frame, self.axis_new_plane, self.new_axis_labels, self.new_axis_colors)



        cv2.imshow("YOLOv5 Tracking", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = YoloV5OpenVINONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
