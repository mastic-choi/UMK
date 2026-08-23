# `signal_state_best_n.pt` 파인튜닝 — Colab 절차 (2026-08-23)

`perception/yolo_signal_state.py`가 쓰는 신호등 색상상태 모델(`red`/`green_straight`/
`green_left`, class id 0/1/2)을 좌회전 데이터로 추가 파인튜닝하는 절차. `yolo_cone.py`/
`yolo_signal.py`와 동일한 export 규약(`imgsz=640, opset=12, simplify=True, nms=True`)을
따른다.

## 0. 데이터셋 준비 완료 (2026-08-23)

새 좌회전 이미지 82장(`green_left`, class 2)에 라벨링이 끝났다(Label Studio류 툴에서
`labels_my-project-name_2026-08-23-11-35-43/` export). 기존 라벨링된 데이터(3클래스,
82장 — train 66/val 16)와 이 새 배치를 **로컬에서 직접 병합**해
`signal_state_dataset_0823.zip`으로 묶어뒀다(Roboflow를 거치지 않음 — 새 배치가 전부
`green_left`뿐이라 자동 split에 맡기지 않고, 기존 train:val ≈ 80:20 비율을 그대로
적용해 새 82장도 66/16으로 나눠 합쳤다):

- `train/`: 149장 — class 분포 red 11 / green_straight 35 / green_left 103
- `valid/`: 36장 — class 분포 red 5 / green_straight 8 / green_left 23
- `data.yaml`: `nc: 3`, `names: ['red', 'green_straight', 'green_left']`

**추가 배치(2026-08-23, 2차):** 첫 재학습(82장 배치)이 런타임 끊김으로 `best.pt`를
날려서 재현하기 전에, 현재 모델을 `좌회전파인튜닝1/2/3`(원본 캡처 6349장, 5장 중 1장
샘플링) 전체에 돌려 `green_left` confidence가 애매하거나(0 < conf < 0.5) 다른 클래스로
오검출한 프레임 155장을 추출했다. 이 중 21장을 실제로 라벨링(전부 `green_left`로
확인됨 — 오검출 의심이었던 것도 실제로는 맞는 검출이었고 모델이 과소확신했던 것)해
위 수치에 반영했다. 나머지 134장은 아직 라벨링 안 됨(향후 배치 후보).

**알려진 한계:** `red`/`green_straight`는 여전히 표본이 적다(train 11/35장) — 이번
파인튜닝은 `green_left` 보강이 목적이므로 의도된 불균형이지만, 학습 후 §4에서
`red`/`green_straight` recall이 떨어지지 않았는지 반드시 확인할 것. 추가 배치 21장은
모델 자신의 예측을 참고해서 "애매한 프레임"만 선별한 것이라(사람이 최종 라벨은
확정했지만) 표본 선정 자체에 모델 편향이 섞여 있을 수 있음.

## 1. Drive 마운트 + 환경 준비

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -q ultralytics
```

## 2. 데이터셋 경로 확인

§0에서 준비한 `signal_state_dataset_0823.zip`을 Drive에 올리고 압축을 풀면 아래 구조가
나온다(`.ipynb` 버전은 Drive 없이 Colab에 직접 업로드 후 `!unzip`으로 그 자리에서 풂):

```yaml
train: ./train/images
val: ./valid/images
nc: 3
names: ['red', 'green_straight', 'green_left']   # config.py YOLO_SIGNAL_STATE_CLASS_NAMES와 순서 반드시 일치
```

```python
DATA_YAML = '/content/drive/MyDrive/UMK/datasets/signal_state_dataset_0823/data.yaml'  # TODO: 실제 경로로 교체
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
