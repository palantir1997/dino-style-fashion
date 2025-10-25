import os
import random
import shutil

# 원본 이미지 경로
source_dir = "/Volumes/archive/images"
# Flask 프로젝트 내 참조 이미지 경로
target_dir = "data/reference_images"
# 무작위로 가져올 이미지 수
sample_count = 500  # 원하는 개수로 조정 가능

# 참조 폴더 생성
os.makedirs(target_dir, exist_ok=True)

# 이미지 파일 리스트
files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 무작위 추출
sampled_files = random.sample(files, min(sample_count, len(files)))

# 복사
for f in sampled_files:
    shutil.copy(os.path.join(source_dir, f), os.path.join(target_dir, f))

print(f"무작위로 {len(sampled_files)}장 복사 완료!")
print(f"저장 경로: {os.path.abspath(target_dir)}")
