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
        description='전방 카메라 장치 경로. xycar_cam.launch.py가 쓰는 usb_cam 기본 params_1.yaml은'
                     ' video_device가 /dev/video0로 고정돼 있어 실제 xycar 카메라(/dev/videoCAM 심볼릭링크)와'
                     ' 안 맞으면 usb_cam_node_exe가 장치를 못 열고 SIGABRT로 죽는다(실측 확인됨).'
                     ' /dev/ttyLIDAR, /dev/ttyIMU와 같은 패턴의 udev 별칭이므로 여기서 직접 지정한다.')

    # ── 센서 드라이버 (카메라/라이다/IMU) ──
    #   YOLO(yolo_ros) 노드는 이 프로젝트에서 더 이상 사용하지 않아 제거함 — 인지는 카메라(차선/신호등)와
    #   라이다(장애물/라바콘)만으로 수행한다.
    #   xycar_cam.launch.py를 include하지 않고 usb_cam_node_exe를 직접 띄우는 이유:
    #   include 방식은 파라미터 오버라이드가 안 되어 video_device를 못 바꾼다.
    #   기본 params_1.yaml(usb_cam 패키지 표준값) 위에 video_device만 덮어써서 실제 장치를 잡는다.
    #   img_left/right/behind는 track_drive.py에서 구독만 하고 실제로 안 쓰이므로 전방 카메라만 띄운다.
    #   'params.yaml'(존재하지 않는 파일명)을 참조하던 버그가 있었다 — usb_cam 패키지엔
    #   params_1.yaml/params_2.yaml만 있어서 launch가 "Parameter file path is not a
    #   file" 경고와 함께 이 파일을 통째로 무시하고 노드 내부 기본값(camera_name=
    #   default_cam)으로 폴백, 존재하지 않는 ~/.ros/camera_info/default_cam.yaml을
    #   찾다가 매 실행마다 에러 로그가 남았다(실측 확인, 2026-08-13). params_1.yaml은
    #   camera_info_url이 usb_cam 패키지에 실제로 있는 config/camera_info.yaml을
    #   가리키므로 이걸로 교체.
    usb_cam_params = os.path.join(
        get_package_share_directory('usb_cam'), 'config', 'params_1.yaml')

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
        # [2026-08-06] 재활성화 — 원래 "imu_yaw는 S2/S3 좌회전 로직에서만 쓰이므로 S0->S1
        # 테스트 단계엔 불필요"라는 이유로 꺼져 있었는데, 이제 imu_yaw가 바퀴 카운트
        # (_update_lap(), 모든 State/Phase에서 동작)와 pure_pursuit의 코너 lookahead 감쇠
        # 보강(README §0.5.5, S1 차선주행에서 항상 씀)에도 쓰여서 그 전제가 더 이상
        # 성립하지 않는다. IMU 하드웨어도 이번에 수리됨(README §8.1) — 다시 켠다.
        imu_include,
        track_drive_node,
    ])
