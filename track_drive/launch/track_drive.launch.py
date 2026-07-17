from launch import LaunchDescription
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
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
        yolo_cone_node,
        yolo_vehicle_node,
        track_drive_node,
    ])
