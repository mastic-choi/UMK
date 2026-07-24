from launch import LaunchDescription
import os

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """수동 조향 데이터 수집(manual_drive_collector.py)용 센서 전용 launch.

    track_drive.launch.py와 달리 track_drive_node(자율주행 FSM)는 띄우지 않는다 —
    같이 띄우면 그쪽도 xycar_motor에 자기 판단대로 명령을 발행해서 수동 조향과
    충돌하기 때문. 카메라(+ 필요시 IMU)만 켠다. 라이다는 이 수집 작업에 필요
    없어서 뺐다.
    """
    video_device = LaunchConfiguration('video_device')
    video_device_cmd = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/videoCAM',
        description='전방 카메라 장치 경로 (track_drive.launch.py와 동일한 이유로 지정)')

    usb_cam_params = os.path.join(
        get_package_share_directory('usb_cam'), 'config', 'params.yaml')

    cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='xycar_cam',
        arguments=['--ros-args', '--log-level', 'error',
                   '--log-level', 'compressed_depth_image_transport:=fatal'],
        parameters=[usb_cam_params, {'video_device': video_device, 'pixel_format': 'yuyv'}],
        remappings=[('image_raw', '/usb_cam/image_raw/front')],
    )

    return LaunchDescription([
        video_device_cmd,
        cam_node,
    ])
