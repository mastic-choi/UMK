
# Odom Subscriber, Quaternion->Yaw, Speed계산 수행
# 현재 차량 위치 관리, global -> local

import math
import numpy as np


class PoseEstimator:


    def global_to_local(
        self,
        points,
        vehicle_x,
        vehicle_y,
        vehicle_yaw
    ):


        R=np.array(
        [
        [math.cos(vehicle_yaw),
        math.sin(vehicle_yaw)],

        [-math.sin(vehicle_yaw),
        math.cos(vehicle_yaw)]
        ]
        )


        local=[]


        for x,y,yaw in points:


            dx=x-vehicle_x
            dy=y-vehicle_y


            p=R@np.array(
                [dx,dy]
            )


            local.append(
                (
                p[0],
                p[1],
                yaw-vehicle_yaw
                )
            )


        return local


#=============================================
# EncoderPoseEstimator — 휠 엔코더(선속도 v, m/s)와 조향각(delta, rad)으로 차량의
# 실세계 pose(x, y, yaw)를 자전거모델 데드레커닝으로 적분 추적한다.
#
# ── 왜 지금 당장 실제 엔코더 토픽에 연결하지 않았나 ──
#   2026-08-05 기준 이 저장소 어디에도 엔코더/오도메트리 ROS 토픽 구독이 없다(전체
#   grep으로 확인함 — track_drive.py는 /usb_cam/*, /scan, /imu만 구독). 실제 로봇에
#   그런 토픽이 퍼블리시되는지, 있다면 이름/타입이 뭔지 아직 확인 전이라 여기서
#   임의로 지어내지 않았다. 이 클래스는 update(v_mps, steer_rad, dt)만 받으면 동작
#   하도록 만들었으니, 실제 토픽을 확인하면:
#     1) track_drive.py에 콜백 추가(cb_encoder 등), 펄스카운트/RPM → m/s 변환
#     2) control_loop() 매 주기 self.pose_estimator.update(v_mps, steer_rad, dt) 호출
#   만 하면 되고 이 클래스 내부는 손댈 필요 없다.
#
# ── yaw를 나중에 IMU로 바꾸는 방법 ──
#   yaw는 기본적으로 조향각 적분(데드레커닝, yaw_dot = v/L*tan(delta))으로 추정하는데,
#   바퀴 슬립·축거 오차가 시간이 지날수록 누적돼 드리프트한다 — pure pursuit에서
#   겪었던 "지그재그가 커지다 한쪽으로 계속 도는" 증상과 같은 계열의 원인이 될 수
#   있다. IMU를 실제로 쓸 수 있게 되면 set_yaw_source('imu')로 전환 — 이후
#   update()에 imu_yaw를 인자로 넘기면 내부 적분 대신 그 값을 그대로 쓰게 바뀐다.
#   x, y 위치 적분(속도만 있으면 됨)은 소스 전환과 무관하게 그대로 유지된다.
#
# ── wheelbase_m 관련 ──
#   controller/pure_pursuit.py의 wheelbase_px와 달리, 이건 픽셀이 아니라 실세계
#   자전거모델에 쓰이는 값이라 카메라 픽셀→미터 환산(PIXELS_PER_METER)과 무관하게
#   그냥 줄자로 앞바퀴-뒷바퀴 축간거리를 재면 된다(별도 캘리브레이션 실험 불필요).
#   미실측 상태(None)면 yaw 적분을 못 하니 경고를 찍고 yaw를 고정한다 — 조용히
#   틀린 값을 내보내지 않는다.
#=============================================
class EncoderPoseEstimator:
    """엔코더 기반 데드레커닝 pose 추정기. planner/hybrid_astar + controller/stanley.py가
    쓰는 self.vehicle_x/y/yaw(track_drive.py)를 대체/보강하는 용도로 설계했다."""

    def __init__(self, wheelbase_m=None):
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.wheelbase_m = wheelbase_m   # 실측 후 대입. None이면 update()가 yaw 적분을 건너뜀.
        self._yaw_source = 'dead_reckoning'   # 'dead_reckoning' | 'imu'
        self._warned_no_wheelbase = False

    def set_yaw_source(self, source):
        assert source in ('dead_reckoning', 'imu'), f"알 수 없는 yaw_source: {source}"
        self._yaw_source = source

    def reset(self, x=0.0, y=0.0, yaw=0.0):
        self.x, self.y, self.yaw = x, y, yaw

    def update(self, v_mps, steer_rad, dt, imu_yaw=None):
        """매 제어주기(track_drive.py control_loop, 20Hz) 호출할 것.
          v_mps     : 엔코더로 잰 선속도(m/s). 후진이면 음수.
          steer_rad : 현재 조향각(rad). 부호 규약은 이 프로젝트 기준
                      (+ = 우회전, track_drive.py TURN_ANGLE=-60이 좌회전인 것과 일치)과
                      일치시켜서 호출할 것.
          dt        : 경과시간(s)
          imu_yaw   : yaw_source='imu'일 때만 전달(rad). track_drive.py의
                      self.imu_yaw를 그대로 넘기면 된다.
        반환 : (x, y, yaw) — 갱신된 pose.
        """
        self.v = v_mps

        if self._yaw_source == 'imu':
            if imu_yaw is None:
                raise ValueError("yaw_source='imu'인데 imu_yaw가 전달되지 않았다")
            self.yaw = imu_yaw
        else:
            if self.wheelbase_m is None:
                if not self._warned_no_wheelbase:
                    print('[EncoderPoseEstimator] 경고: wheelbase_m 미실측 — yaw 적분 불가, yaw를 고정값으로 유지')
                    self._warned_no_wheelbase = True
            else:
                yaw_rate = (v_mps / self.wheelbase_m) * math.tan(steer_rad)
                self.yaw += yaw_rate * dt

        self.x += v_mps * math.cos(self.yaw) * dt
        self.y += v_mps * math.sin(self.yaw) * dt

        return self.x, self.y, self.yaw
