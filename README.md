# dino/clip style fashion 
<iframe width="1170" height="626"
  src="https://www.youtube.com/embed/J3t5VXPHOt4"
  title="DINO/CLIP Fashion Demo"
  frameborder="0"
  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
  allowfullscreen>
</iframe>

---

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

![Image](https://github.com/user-attachments/assets/0b85cb49-e035-4a07-a9ee-d10c9a80ef2f)
![Image](https://github.com/user-attachments/assets/25bc14fa-84fc-48cb-8cb6-f0270e3a13eb)

---

### Python API로 학습

#### 파이썬 API로 실행 
```
>>> python app.py
→ 브라우저에서 http://127.0.0.1:5000 접속
```

---

#### 실행결과
<img width="895" height="798" alt="Image" src="https://github.com/user-attachments/assets/5f2f8b93-da2e-4cf1-8d8c-6afd1e594d7b" />
<img width="974" height="624" alt="Image" src="https://github.com/user-attachments/assets/4a0571af-b186-4e58-a141-fd0bb26a5904" />

---




