from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='mmdet3d_ros2',
            executable='infer_node',
            name='mmdet3d_infer_node',
            parameters=[
                {'config_file': '/home/chenx2/mmdetection3d/projects/TR3D/configs/tr3d_1xb16_sunrgbd-3d-10class.py'},
                {'checkpoint_file': '/home/chenx2/checkpoints/tr3d_1xb16_sunrgbd-3d-10class.pth'},
                {'score_threshold': 0.01},
                {'infer_device': 'cuda:0'},
                {'nms_interval': 0.5},
                {'point_cloud_qos': 'reliable'},
                {'point_cloud_topic': '/femto_mega/depth_registered/filter_points'}
            ]
        )
    ])