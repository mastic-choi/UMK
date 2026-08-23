# `signal_state_best_n.pt` 파인튜닝 — Colab 절차 (2026-08-23)

`perception/yolo_signal_state.py`가 쓰는 신호등 색상상태 모델(`red`/`green_straight`/
`green_left`, class id 0/1/2)을 좌회전 데이터로 추가 파인튜닝하는 절차. `yolo_cone.py`/
`yolo_signal.py`와 동일한 export 규약(`imgsz=640, opset=12, simplify=True, nms=True`)을
따른다.

## 0. 먼저 확인할 것 — 새 이미지는 아직 라벨이 없음

`대외활동/yolo딥러닝/신호등/0823_좌회전_선택/`에 골라둔 86장은 **bbox 라벨이 없는
이미지**다. Roboflow(또는 CVAT/LabelImg)로 `green_left`(class 2) bbox를 먼저 그려야
한다 — 신호등판/화살표 램프 부분을 감싸는 박스 1개씩, 기존 `datasets/signal_state/`
라벨링 방식과 동일하게.

**기존 데이터와 반드시 합쳐서 학습할 것 — 새 이미지만 따로 학습하지 말 것.** 새
86장은 전부 `green_left`뿐이라, 이것만 학습하면 `red`/`green_straight`를 모델이
잊어버리는 catastrophic forgetting 위험이 크다. Roboflow에서 기존 프로젝트에 이
86장을 새 배치로 추가하는 방식을 권장(그러면 train/val split도 자동으로 섞임).

## 1. Drive 마운트 + 환경 준비

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics
```

## 2. 데이터셋 경로 확인

기존 `datasets/signal_state/`(Roboflow export)에 새 86장 배치를 합친 뒤 내보낸
`data.yaml`이 아래 형태인지 확인:

```yaml
train: /content/drive/MyDrive/.../signal_state/train/images
val: /content/drive/MyDrive/.../signal_state/valid/images
nc: 3
names: ['red', 'green_straight', 'green_left']   # config.py YOLO_SIGNAL_STATE_CLASS_NAMES와 순서 반드시 일치
```

```python
DATA_YAML = '/content/drive/MyDrive/UMK/datasets/signal_state/data.yaml'  # TODO: 실제 경로로 교체
```

## 3. 기존 체크포인트에서 파인튜닝 (처음부터 재학습 아님)

`yolov8n.pt`부터 새로 학습하지 않고, 기존에 학습해둔 `signal_state_best_n.pt`
가중치를 불러와 이어서 학습한다 — 이미 배운 `red`/`green_straight`는 그대로 유지한 채
`green_left` 표현력만 보강하는 게 목적이므로, epoch은 적게 + lr은 낮게 잡는다.

```python
from ultralytics import YOLO

BASE_CKPT = '/content/drive/MyDrive/UMK/yolo_ros/signal_state_best_n.pt'  # TODO: 기존 .pt 경로
model = YOLO(BASE_CKPT)

results = model.train(
    data=DATA_YAML,
    epochs=30,            # 처음 학습 때보다 짧게 — 이미 수렴한 모델의 미세조정
    imgsz=640,
    batch=16,
    lr0=0.001,             # 기본값(0.01)보다 낮춰 기존 클래스 붕괴 방지
    patience=10,           # val 성능 정체 시 조기종료
    project='signal_state_finetune',
    name='left_boost_0823',
    exist_ok=True,
)
```

학습 후 반드시 확인할 것: `results` 또는 `runs/.../results.csv`의 **클래스별**
precision/recall — `green_left`만 오르고 `red`/`green_straight`가 떨어졌다면 lr을 더
낮추거나 epoch을 줄여 재시도.

## 4. 검증 — 클래스별 confusion matrix

```python
metrics = model.val(data=DATA_YAML)
print(metrics.box.maps)       # 클래스별 mAP50-95: [red, green_straight, green_left] 순
```

`runs/detect/.../confusion_matrix.png`도 같이 열어서 `green_left`가 실제로
`green_straight`나 배경과 헷갈리고 있는지 눈으로 확인 — 이전 대화에서 다룬
"좌회전 신뢰도가 낮은 원인"이 데이터 부족이었는지 시각적 유사성 문제였는지 여기서 감별된다.

## 5. ONNX export — 기존 규약과 동일

```python
best_pt = model.trainer.best  # runs/.../weights/best.pt
best_model = YOLO(best_pt)
best_model.export(format='onnx', imgsz=640, opset=12, simplify=True, nms=True)
```

## 6. 결과물 반영

```python
import shutil
shutil.copy(str(best_pt).replace('.pt', '.onnx'),
            '/content/drive/MyDrive/UMK/yolo_ros/signal_state_best_n.onnx')
```

Drive에서 받은 새 `signal_state_best_n.onnx`를 저장소 `yolo_ros/signal_state_best_n.onnx`에
덮어쓰고, `track_drive/track_drive/README.md`에 이번 파인튜닝 배경/결과(§1.16 계열)를
남긴 뒤 실차에서 `DEBUG_VIZ_YOLO_SIGNAL_STATE=True`로 기존 Hough 판정과 비교 검증할 것 —
저장소 관례상(§ CLAUDE.md) 실차 미검증 상태로는 `perc_signal()` 판단 소스로 바꾸지 않는다.
