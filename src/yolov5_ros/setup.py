from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cuon',
    maintainer_email='anhquan.tran203@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_node = yolov5_ros.yolov5_node:main',
            'camera_test = yolov5_ros.camera_test:main',
            'img_publisher = yolov5_ros.webcam_pub:main',
            'img_subscriber = yolov5_ros.webcam_sub:main',
        ],
    },
)
