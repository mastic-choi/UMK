from launch import LaunchDescription
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource

# ── YOLO 가중치 기본 경로 ──
# .pt 파일은 패키지로 설치되지 않으므로 소스 트리(yolo_ros/) 경로를 직접 가리킨다.
# 실차 워크스페이스 위치가 다르면 launch 인자(cone_model/vehicle_model)로 덮어쓸 것.
YOLO_WEIGHTS_DIR = os.path.expanduser('~/xycar_ws/src/yolo_ros')


def generate_launch_description():

    # ── YOLO 이중확인용 노드 2개 (Phase별 역할 분담, enable 서비스로 전환) ──
    #   yolo_cone    : 라바콘 파인튜닝 모델(cone_best.pt) — 시작 Phase가 LAVACON이므로 켠 상태로 시작
    #   yolo_vehicle : COCO 사전학습 모델(yolov8m.pt)     — 꺼둔 상태로 대기, 라바콘 종료 시점에 켬
    #   전환은 track_drive.py가 /yolo_cone/enable, /yolo_vehicle/enable 서비스(SetBool)로 수행한다.
    cone_model = LaunchConfiguration('cone_model')
    cone_model_cmd = DeclareLaunchArgument(
        'cone_model',
        default_value=os.path.join(YOLO_WEIGHTS_DIR, 'cone_best.pt'),
        description='라바콘 전용 파인튜닝 YOLO 모델(.pt) 경로')

    vehicle_model = LaunchConfiguration('vehicle_model')
    vehicle_model_cmd = DeclareLaunchArgument(
        'vehicle_model',
        default_value=os.path.join(YOLO_WEIGHTS_DIR, 'yolov8m.pt'),
        description='방해차량 인식용 COCO 사전학습 YOLO 모델(.pt) 경로')

    device = LaunchConfiguration('device')
    device_cmd = DeclareLaunchArgument(
        'device',
        default_value='cpu',  # Jetson에서 CUDA torch 설치 시 'cuda:0'으로 덮어쓸 것
        description='YOLO 추론 디바이스 (cpu / cuda:0)')

    cone_threshold = LaunchConfiguration('cone_threshold')
    cone_threshold_cmd = DeclareLaunchArgument(
        'cone_threshold',
        default_value='0.4',
        description='라바콘 검출 최소 신뢰도 (라이다 AND 결합이라 다소 낮게)')

    vehicle_threshold = LaunchConfiguration('vehicle_threshold')
    vehicle_threshold_cmd = DeclareLaunchArgument(
        'vehicle_threshold',
        default_value='0.3',
        description='차량 검출 최소 신뢰도 (실측 truck 0.40~0.71 + 실차 웹캠 화질 감안해 낮게)')

    use_debug = LaunchConfiguration('use_debug')
    use_debug_cmd = DeclareLaunchArgument(
        'use_debug',
        default_value='true',
        description='YOLO 검출 시각화(debug_node) 실행 여부 — dbg_image 토픽을 rqt_image_view로 확인')

    video_device = LaunchConfiguration('video_device')
    video_device_cmd = DeclareLaunchArgument(
        'video_device',
        default_value='/dev/videoCAM',
        description='전방 카메라 장치 경로. xycar_cam.launch.py가 쓰는 usb_cam 기본 params.yaml은'
                     ' video_device가 /dev/video0로 고정돼 있어 실제 xycar 카메라(/dev/videoCAM 심볼릭링크)와'
                     ' 안 맞으면 usb_cam_node_exe가 장치를 못 열고 SIGABRT로 죽는다(실측 확인됨).'
                     ' /dev/ttyLIDAR, /dev/ttyIMU와 같은 패턴의 udev 별칭이므로 여기서 직접 지정한다.')

    # ── 센서 드라이버 (카메라/라이다/IMU) ──
    #   이 launch는 원래 YOLO+track_drive만 띄우고 있었고 센서 드라이버가 빠져 있었다
    #   (실차에서 img_front/lidar_ranges가 전혀 안 들어와 S0에서 계속 멈춰있던 원인).
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

    yolo_cone_node = Node(
        package='yolo_ros',
        executable='yolo_node',
        name='yolo_node',
        namespace='yolo_cone',
        parameters=[{
            'model': cone_model,
            'device': device,
            'enable': True,                 # Phase.LAVACON부터 시작하므로 켠 상태
            'threshold': cone_threshold,
            'image_reliability': 1,         # usb_cam(RELIABLE)과 QoS 일치
        }],
        remappings=[('image_raw', '/usb_cam/image_raw/front')],
    )

    yolo_vehicle_node = Node(
        package='yolo_ros',
        executable='yolo_node',
        name='yolo_node',
        namespace='yolo_vehicle',
        parameters=[{
            'model': vehicle_model,
            'device': device,
            'enable': False,                # 라바콘 구간 종료 시 track_drive가 켬
            'threshold': vehicle_threshold,
            'image_reliability': 1,
        }],
        remappings=[('image_raw', '/usb_cam/image_raw/front')],
    )

    # ── YOLO 디버그 시각화 (검출 bbox를 원본 이미지에 그려 dbg_image로 발행) ──
    #   확인: ros2 run rqt_image_view rqt_image_view → /yolo_cone/dbg_image, /yolo_vehicle/dbg_image 선택
    #   비활성화된 yolo_node는 detections를 발행하지 않으므로 해당 dbg_image도 멈추는 것이 정상.
    yolo_cone_debug_node = Node(
        package='yolo_ros',
        executable='debug_node',
        name='debug_node',
        namespace='yolo_cone',
        parameters=[{'image_reliability': 1}],
        remappings=[('image_raw', '/usb_cam/image_raw/front')],
        condition=IfCondition(use_debug),
    )

    yolo_vehicle_debug_node = Node(
        package='yolo_ros',
        executable='debug_node',
        name='debug_node',
        namespace='yolo_vehicle',
        parameters=[{'image_reliability': 1}],
        remappings=[('image_raw', '/usb_cam/image_raw/front')],
        condition=IfCondition(use_debug),
    )

    track_drive_node = Node(
        package='track_drive',
        executable='track_drive',
        name='driver',
        parameters=[{'speed': 12}],
    )

    return LaunchDescription([
        cone_model_cmd,
        vehicle_model_cmd,
        device_cmd,
        cone_threshold_cmd,
        vehicle_threshold_cmd,
        use_debug_cmd,
        video_device_cmd,
        cam_node,
        lidar_include,
        # imu_include,  # S0->S1 테스트 단계에서 비활성화. imu_yaw는 S2/S3 좌회전 로직에서만 쓰이므로
        #               (track_drive.py의 _yaw_delta/_begin_left_turn/_s3_shortcut) 지금은 불필요.
        #               좌회전(S2/S3) 테스트 시작하면 이 줄 다시 살릴 것.
        yolo_cone_node,
        yolo_vehicle_node,
        yolo_cone_debug_node,
        yolo_vehicle_debug_node,
        track_drive_node,
    ])
