from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Webcam publisher node
        Node(
            package='yolov5_ros',
            executable='img_publisher',
            name='webcam_publisher',
            output='screen',
            remappings=[
                ('/camera/image_raw', '/camera/image_raw')
            ]
        ),

        # YOLOv5 detection node
        Node(
            package='yolov5_ros',
            executable='yolov5_node',
            name='yolov5_node',
            output='screen',
            parameters=[
                {'model_path': '/home/cuon/yolov5_ros2_ws/src/yolov5_ros/models/yolov5s.xml'},
                {'confidence_threshold': 0.25},
                {'iou_threshold': 0.45}
            ],
            remappings=[
                ('/image_raw', '/camera/image_raw') 
            ]
        )
    ])
