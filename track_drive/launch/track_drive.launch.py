from launch import LaunchDescription
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():

    video_device = LaunchConfiguration('video_device')
    video_device_cmd = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/videoCAM',
        description='전방 카메라 장치 경로. xycar_cam.launch.py가 쓰는 usb_cam 기본 params.yaml은'
                     ' video_device가 /dev/video0로 고정돼 있어 실제 xycar 카메라(/dev/videoCAM 심볼릭링크)와'
                     ' 안 맞으면 usb_cam_node_exe가 장치를 못 열고 SIGABRT로 죽는다(실측 확인됨).'
                     ' /dev/ttyLIDAR, /dev/ttyIMU와 같은 패턴의 udev 별칭이므로 여기서 직접 지정한다.')

    # ── 센서 드라이버 (카메라/라이다/IMU) ──
    #   YOLO(yolo_ros) 노드는 이 프로젝트에서 더 이상 사용하지 않아 제거함 — 인지는 카메라(차선/신호등)와
    #   라이다(장애물/라바콘)만으로 수행한다.
    #   xycar_cam.launch.py를 include하지 않고 usb_cam_node_exe를 직접 띄우는 이유:
    #   include 방식은 파라미터 오버라이드가 안 되어 video_device를 못 바꾼다.
    #   기본 params.yaml(usb_cam 패키지 표준값) 위에 video_device만 덮어써서 실제 장치를 잡는다.
    #   img_left/right/behind는 track_drive.py에서 구독만 하고 실제로 안 쓰이므로 전방 카메라만 띄운다.
    usb_cam_params = os.path.join(
        get_package_share_directory('usb_cam'), 'config', 'params.yaml')

    cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='xycar_cam',
        # compressed_depth_image_transport: usb_cam이 image_transport_plugins에 의존하다보니
        # 뎁스가 아닌 일반 컬러 프레임에도 뎁스 압축을 시도하다 실패 로그를 매 프레임 찍는다
        # (기능상 무해 — 아무도 .../compressedDepth 토픽을 구독하지 않음). 해당 로거만 조용히 시킴.
        arguments=['--ros-args', '--log-level', 'error',
                   '--log-level', 'compressed_depth_image_transport:=fatal'],
        # pixel_format 오버라이드: 기본값(mjpeg2rgb, avcodec 디코드)이 이 카메라의 640x480/30fps
        # 모드와 협상 실패해 usb_cam_node_exe가 시작 직후 char* 예외로 죽는 문제(SIGABRT) 발견됨.
        # v4l2-ctl --list-formats-ext 확인 결과 이 카메라는 640x480을 YUYV로 30fps 네이티브 지원하므로
        # avcodec 디코드 경로를 타지 않는 yuyv로 강제 지정한다.
        parameters=[usb_cam_params, {'video_device': video_device, 'pixel_format': 'yuyv'}],
        remappings=[('image_raw', '/usb_cam/image_raw/front')],
    )
    lidar_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('xycar_lidar'), 'launch', 'xycar_lidar.launch.py'))
    )
    imu_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('xycar_imu'), 'launch', 'xycar_imu.launch.py'))
    )

    track_drive_node = Node(
        package='track_drive',
        executable='track_drive',
        name='driver',
        parameters=[{'speed': 12}],
    )

    return LaunchDescription([
        video_device_cmd,
        cam_node,
        lidar_include,
        # imu_include,  # S0->S1 테스트 단계에서 비활성화. imu_yaw는 S2/S3 좌회전 로직에서만 쓰이므로
        #               (track_drive.py의 _yaw_delta/_begin_left_turn/_s3_shortcut) 지금은 불필요.
        #               좌회전(S2/S3) 테스트 시작하면 이 줄 다시 살릴 것.
        track_drive_node,
    ])
