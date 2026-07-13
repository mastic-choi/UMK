
import rospy, time, math, os

from lane_util import CameraProcessor, SlideWindow



# 변수
bridge = CvBridge()
image = np.empty(shape=[0])
WIDTH, HEIGHT = 640, 480 
Blue = (255,0,0)
Green = (0,255,0)
Red = (0,0,255)
Yellow = (0,255,255)
View_Center = WIDTH//2 # 화면의 중앙값 = 카메라 위치

# 콜백 함수 - USB카메라 토픽을 받아 처리
def usbcam_callback(data):
    global image
    image = bridge.image_to_cv2(data, "bgr8")

def compressed_image_callback(data):
    global compressed_image
    np_arr = np.frombuffer(data.data, np.unit8)
    compressed_image = cv2.imdecode(np_arr, cv2.IMAGED_COLOR)


# 정지선 확인 후 True/False 반환

def check_stopline():
    global stopline_num

    #image 잘라내기(ROI Area)
    roi_img = image[270:320, 0:640]
    cv2.imshow("ROI Image", roi_img)

    #HSV 변환, V채널에 대해 범위를 정해 흑백 이진화 이미지로 변환
    hsv_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
    upper_white = np.array([255, 255, 255])
    lower_white = np.array([0,0,100])
    binary_img = cv2.inRange(hsv_image, lower_white, upper_white)

    #흑백이진화 이미지에서 정지선 체크용 이미지 만들기
    stopline_check_img = binary_img[0:50, 150:480]

    #컬러 이미지로 바꾼 후 정지선 체크용 이미지 영역을 녹색 사각형으로 표시
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (200,100),(440,120),Green,3)
    cv2.imshow('Stopline Check', img)
    cv2.waitKey(1)

    #정지선 체크용 이미지에서 흰색 점의 개수 카운트
    stopline_count = cv2.countNonZero(stopline_check_img)

    #사각형 안의 흰색 점이 기준치 이상이면 True
    if stopline_count > 1000:
        print("Stopline Found...! -", stopline_num)
        stopline_num = stopline_num + 1
        return True
    
    else:
        return False
    

            
class LaneDetector:
    def __init__(self, camera_processor=None, slide_window_processor=None):
        self.camera_processor = camera_processor
        self.slide_window_processor = slide_window_processor

    def set_processor(self, camera, slide_window):
        self.camera_processor = camera
        self.slide_window_processor = slide_window

    def detect(self, frame):
        #CameraProcessor 에서 BEV, mask 생성
        bev, white_mask, yellow_mask = self.camera_processor.processor(frame)

        if bev is None:
            return False, 0,0, 0,0, None
        
        lane_valid, lane_offset, lookahead = self.slide_window_processor.detect(
            bev, white_mask, yellow_mask
        )

        return lane_valid, lane_offset, lookahead, bev
    
            
# 메인 함수
def start():

    camera_processor = CameraProcessor()
    slide_window_processor = slideWindow()

    lane_detector = LaneDetector(
        camera_processor, slide_window_processor
    )

    #모드 변경 상수 아래에 적기
    HSV = 11
    #어떤 미션부터 수헹할 것인지 결정
    drive_mode = HSV
    cam_exposure(100)

    #노드 생성, 구독/발행할 토픽 선언
    rospy.init_node('Track_Driver')
    #...
    
    #발행자 노드들로부터 첫번째 토픽이 도착할때까지 대기
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready------------")
    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("UltraSonic Ready------------")
    rospy.wait_for_message("/scan", LaserScan)
    print("Lidar Readt------------")

    print("===================================")
    print(" S T A R T    D R I V I N G . . .")

    #main loop
    camera_processor = CameraProcessor()
    slide_window_processor = SlideWindow()
    lane_detector.set_processor(camera_processor, slide_window_processor)
    perv_data = {}


