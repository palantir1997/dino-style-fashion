# dino/clip style fashion 

## 프로젝트 개요
**DINO 모델을 활용한 패션 이미지 유사도 추천 웹 서비스**. <br>
이 프로젝트는 DINOv2와 CLIP 모델을 활용하여 사용자가 업로드한 패션 이미지와 시각적 또는 의미적으로 유사한 스타일의 이미지를 찾아주는 웹 애플리케이션입니다.

---

### 데이터셋 출처 및 모델 정보

| 구분 | 모델 이름 | 모델 출처 | 주요 역할 | 사용 목적 |
|------|------------|------------|-------------|-------------|
| **이미지 임베딩 (시각적 특징 추출)** | **DINOv2 (facebook/dinov2-base)** | Meta AI | 이미지의 시각적 특징을 벡터(숫자) 형태로 추출 | 업로드된 이미지와 기존 데이터셋 이미지 간의 **시각적 유사도 계산용** |
| **이미지-텍스트 임베딩 (의미 기반 유사도)** | **CLIP (openai/clip-vit-base-patch32)** | OpenAI | 이미지와 텍스트를 같은 임베딩 공간에 매핑 | **의미적으로 비슷한 이미지 검색** (예: "셔츠" 비슷한 이미지 찾기) |
| **이미지 캡션 생성** | **BLIP (Salesforce/blip-image-captioning-base)** | Salesforce Research | 이미지 내용을 자연어 문장으로 설명 | 업로드된 이미지나 추천 이미지에 대해 **자연스러운 설명문 생성** |
| **벡터 데이터베이스** | **ChromaDB** | Chroma | DINO/CLIP 임베딩 벡터를 저장 및 인덱싱 | 유사도 검색 시 빠른 벡터 검색 수행 |
| **일반 데이터베이스** | **MongoDB** | — | 원본 이미지 파일 및 메타데이터 저장 | 실제 이미지 데이터 관리 |
| **웹 애플리케이션** | **Flask Web Framework** | — | 사용자 인터페이스 및 서버 처리 | 이미지 업로드 → 임베딩 → 검색 → 결과 표시 |

- 이미지 데이터셋 출처: [kaggle](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small)

---

## 설치 및 환경 설정
프로젝트를 실행하려면 다음 패키지들을 설치해야 합니다.

### 시스템 가상환경

```
- OS: macOS (M1 칩)
- GPU: 사용 가능 (Apple Silicon GPU / Metal 지원)
- Python 버전: 3.10.18
- 가상환경: miniconda 기반
```

### MongoDB 설치 생성 및 활성화
```
Homebrew로 MongoDB 설치 후, MongoDB 서비스 시작
brew services start mongodb-community
```

### 필요한 라이브러리 설치
```
pip install torch transformers pillow flask pymongo chromadb
```

---

### 구조

```

# DINO 모델을 이용해 업로드한 옷 이미지와 유사한 옷을 추천해주는 Flask 웹서비스
# mongoDb 터미널에서 실행 -> brew services start mongodb-community

dino-style-fashion/
│
├── app.py                        
│   └─ Flask 웹 서버의 메인 진입점
│      - 이미지 업로드 처리
│      - DINO 모델과 DB를 연결
│      - 추천 결과를 HTML로 렌더링
│
├── templates/
│   └── index.html                
│      - 메인 웹페이지 UI
│      - 이미지 업로드 폼, 추천 결과 표시
│
├── static/
│   ├── uploads/                  
│      - 사용자가 업로드한 이미지 저장 폴더
│   ├── recommend/                  
│      - 추천 결과 이미지를 임시 저장
│
├── models/
│   └── dino_model.py             
│      - DINO 모델 불러오기
│      - 이미지 임베딩(벡터) 추출 기능 제공
|   ├── clip_model.py (이미지 캡션 기능)
│
├── utils/
│   └── chroma_utils.py           
│      - ChromaDB 초기화 및 관리
│      - 벡터 유사도 검색 함수 포함
│
├── database/
│   └── mongo_db.py               
│      - MongoDB 연결 설정
│      - 이미지 메타데이터 저장 / 조회 함수
│
├── scripts/
│   ├── init_reference_images.py  
│      - 초기 reference 이미지 등록 스크립트
│   ├── check_chroma.py           
│      - ChromaDB 데이터 점검용
│   └── debug_chroma.py           
│      - 유사도 검색 디버깅용 (테스트 실행)
│
└── data/
    └── reference_images/      
       - reference_images.py    
       - 비교용 원본 이미지 데이터 저장소


# 실제 구현 되는 서비스 설계도
[사용자 업로드]
        │
        ▼
templates/index.html
        │
        ▼
app.py
 ├─ (1) 업로드 이미지 저장 → static/uploads/
 ├─ (2) DINO 모델 호출 (models/dino_model.py)
 │      └─ 이미지 임베딩 추출
 ├─ (3) ChromaDB 검색 (utils/chroma_utils.py)
 │      └─ 유사도 높은 이미지의 mongo_id 리스트 반환
 ├─ (4) MongoDB 조회 (database/mongo_db.py)
 │      └─ 해당 mongo_id로 실제 이미지 가져오기
 │      └─ static/recommend/{mongo_id}.jpg 로 임시 저장
 └─ (5) 결과를 index.html 로 전달 (render_template)
        │
        ▼
templates/index.html
 └─ 업로드 이미지 + 추천 결과 이미지 렌더링


```

### YAML

```
train: C:/Users/User/your_url/Fish-breeds/train/images
val: C:/Users/User/your_url/Fish-breeds/valid/images
test: C:/Users/User/your_url/Fish-breeds/test/images
Roboflow URL: https://universe.roboflow.com/licenta-ldbud/fish-breeds-aeg36/dataset/9
```

---

## 모델 학습
YOLOv8 모델을 학습시키는 방법은 두 가지가 있습니다: CLI(명령어 창) 방식과 Python API 방식.

### Python API로 학습

#### 파이썬 API로 실행 default(epoch15) 테스트
```
>>> python
>>> model = YOLO(r"C:\Users\User\your_url\yolov8n.pt")
>>> result = model.train(data=r"C:\Users\User\your_url\data.yaml", epochs=15)
```

#### scratch(epochs=30 batch=16) 테스트
```
>>> python
>>> model = YOLO(r"C:\Users\User\your_url\yolov8n.pt")
>>> result = model.train(data='data.yaml', epochs=30, batch=16)
```

#### freeze(epochs=30 freeze=10 batch=16) 테스트
```
>>> python
>>> model = YOLO(r"C:\Users\User\your_url\yolov8n.pt")
>>> result = model.train(data=r"C:\Users\User\your_url\data.yaml", epochs=30, freeze=10, batch=16)
```

#### 3개 데이터 모델 결과
<img width="2245" height="1604" alt="Image" src="https://github.com/user-attachments/assets/91d58423-e695-4bc5-bc8f-786d48c27ce3" />

#### 마지막 freeze 테스트 결과
<img width="3000" height="2250" alt="Image" src="https://github.com/user-attachments/assets/8facdcab-bf2f-4641-9249-9f6af8472fe4" />

<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/3410c8a2-c301-4853-8cc5-a5b29be14b23" />

---

## 모델 추론 및 사용
학습이 완료되면 runs/detect/train/weights/best.pt 경로에 최적의 모델 가중치 파일이 저장됩니다. 이 파일을 사용해 이미지, 동영상 등 다양한 소스에 대한 객체 탐지를 수행할 수 있습니다.

### 1. 이미지 또는 폴더에 대한 추론

```
- 1. 학습된 모델 로드(필자는 13, 15, 17에 각각 학습시켰던 결과가 runs파일안에 있었음)
>>> model = YOLO(r"C:/Users/User/runs/detect/train15/weights/best.pt")

- 2. 이미지 폴더에 대한 예측 수행 및 결과 저장
>>> results = model.predict(source=r"C:/Users/User/your_url/Fish-breeds/test/images", save=True, conf=0.25) # 임계점 0.25

- 3. 결과 하나 확인
>>> results[0].show()

- 여러개 확인하고 싶을 시 for문 사용

```

### 2. 동영상 파일에 대한 추론

```
- 학습된 모델 로드
>>> model = YOLO(r"C:/Users/User/runs/detect/train15/weights/best.pt")

- 비디오 파일 경로
>>> video_path = r"C:\Users\User\your_url\test_video\fish2.mp4"

- 비디오에 대한 예측 수행 및 결과 저장
>>> results = model.predict(source=video_path, save=True, conf=0.25)

```

###  3. OpenCV와 연동하여 실시간 추론 및 시각화
```
OpenCV를 사용하면 비디오 프레임별로 예측 결과를 시각적으로 표시할 수 있습니다. 경계 상자 외에 물고기의 중심점을 표시하고 클래스 이름 텍스트를 추가하는 기능을 포함합니다.

from ultralytics import YOLO
import cv2
model = YOLO(r"C:\Users\User\runs\detect\train15\weights\best.pt")
cap = cv2.VideoCapture(r"C:\Users\User\your_url\test_video\fish2.mp4")
frameCount = 0
while (True) :
    ret, frame = cap.read()
    if (not(ret)) : break
    
    frame = cv2.resize(frame, dsize=(640, 360))
    result = model.predict(source=frame, show=True, verbose=False, stream=False, conf=0.7, imgsz=640)
    res = result[0]  # cap.read()에서 한장의 이미지 프레임만 읽어 예측했기 때문에, result[0]

    for box in res.boxes :
        print(f"FrameCount = {frameCount}, {box.data.cpu().numpy()}")
        npp = box.xyxy.cpu().numpy()
        npcls = box.cls.cpu().numpy()
        cx = int((npp[0][0]+npp[0][2])/2)
        cy = int((npp[0][1]+npp[0][3])/2)
        frame = cv2.circle(frame, (cx, cy),30, (0,255,255), 3)
        cv2.putText(frame, res.names[npcls[0]] , org=(cx,cy), color=(0,255,0), fontScale=1, thickness=2, lineType = cv2.LINE_AA, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        cv2.imshow('Detected Object', frame) 
        if (cv2.waitKey(1) == 27) :
            break
        frameCount += 1
cap.release()
cv2.destroyAllWindows()

```



