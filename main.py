import streamlit as st
import os
from PIL import Image
import numpy as np
import albumentations as A
import shutil

# 증강 목록 정의
AUGMENTATIONS = {
    # 회전: limit 각도 범위 설정 (양수/음수 회전)
    "rotate": A.Rotate(limit=25, p=1.0),  # ← limit를 90으로 하면 이미지가 90도 회전됨

    # 기울이기(Shear): 좌우/상하로 찌그러뜨리는 효과
    "shear": A.Affine(shear=(-16, 16), p=1.0),  # ← shear 범위를 넓히면 더 찌그러짐

    # 수직 뒤집기
    "vertical-flip": A.VerticalFlip(p=1.0),  # ← 이미지를 위아래로 뒤집음

    # 수평 뒤집기
    "horizontal-flip": A.HorizontalFlip(p=1.0),  # ← 이미지를 좌우로 뒤집음

    # 중앙 기준 좌우로 확대: 실제 자르지 않고 비율만 늘리는 효과
    "crop": A.Affine(
        scale={"x": 1.3, "y": 1.0},  # ← x 값 키우면 좌우로 늘어남, y는 세로 크기
        translate_percent={"x": 0.0, "y": 0.0},  # 중심 고정
        fit_output=True,
        p=1.0
    ),

    # 자르지 않고 주변에 패딩 추가 (반사 방식)
    "crop-and-pad": A.PadIfNeeded(min_height=256, min_width=256, border_mode=4, p=1.0),
    # border_mode: 0 → 검정, 4 → 반사

    # 원근 왜곡 (3D 느낌의 찌그러짐)
    "perspective-transform": A.Perspective(scale=(0.05, 0.1), p=1.0),  # ← scale 높이면 왜곡이 커짐

    # 탄성 변형 (물결처럼 찌그러짐)
    "elastic-transformation": A.ElasticTransform(alpha=200, sigma=25, p=1.0),  # ← alpha 크면 변형이 강해짐

    # 선명하게 (강한 밝기와 경계 강조)
    "sharpen": A.Sharpen(
        alpha=(0.8, 1.0),       # ← 선명도 마스크 강도
        lightness=(1.5, 2.0),   # ← 밝은 영역 강조 → 밝을수록 눈부심 느낌
        p=1.0
    ),

    # 밝게 만들기
    "brighten": A.RandomBrightnessContrast(
        brightness_limit=(0.4, 0.6),  # ← 밝기만 조절
        contrast_limit=(0.0, 0.1),
        p=1.0
    ),

    # 감마 대비 조정 (어두운 톤)
    "Gamma-contrast": A.RandomGamma(
        gamma_limit=(30, 50),  # ← 낮은 gamma → 전체적으로 어두운 톤
        p=1.0
    ),

    # 색 반전
    "invert": A.InvertImg(p=1.0),  # ← RGB를 반전 (흰 → 검, 검 → 흰)

    # 블러 처리 (흐리게)
    "gaussian-blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),  # ← blur_limit이 클수록 더 흐려짐

    # 가우시안 노이즈 추가
    "additive-gaussian-noise": A.GaussNoise(p=1.0),  # ← 픽셀마다 랜덤 노이즈 추가됨

    # 수평 왜곡 (글리치처럼)
    "translate-x": A.GridDistortion(num_steps=10, distort_limit=0.2, p=1.0),  # ← 줄 단위로 흔들리는 느낌

    # 수직 왜곡 (물결 흐림)
    "translate-y": A.OpticalDistortion(distort_limit=0.5, p=1.0),  # ← 왜곡량 조정

    # coarse-salt: 흰 사각형 블럭 무작위로 배치 (노이즈 느낌)
    "coarse-salt": A.CoarseDropout(
        min_holes=15, max_holes=20,      # ← 블럭 개수
        min_height=10, max_height=20,    # ← 블럭 크기
        min_width=10, max_width=20,
        fill_value=255,                  # ← 255: 흰색, 0: 검정
        p=1.0
    ),

    # 해상도 낮추기 → 픽셀화 느낌
    "super-pixel": A.Downscale(
        scale_min=0.1,
        scale_max=0.2,  # ← 값 작을수록 더 뭉개짐
        p=1.0
    ),

    # 엠보스 + 색 강조
    "emboss": A.Compose([
        A.Emboss(alpha=(0.8, 1.0), strength=(0.8, 1.0), p=1.0),  # ← 윤곽 강조
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),  # ← 대비 보정
        A.HueSaturationValue(  # ← 색 강조
            hue_shift_limit=10,
            sat_shift_limit=30,
            val_shift_limit=10,
            p=1.0
        )
    ]),

    # 구름 느낌 (희뿌연 흐림) + 압축 아티팩트
    "clouds": A.Compose([
        A.RandomFog(p=1.0),  # ← 흐릿하게
        A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0)  # ← 압축 손상 느낌
    ]),

    # 안개 효과만
    "fog": A.RandomFog(p=1.0),  # ← 흐릿한 뿌연 느낌

    # 눈 내리는 효과 (점, 선 형태)
    "snow-flakes": A.RandomSnow(
        brightness_coeff=2.5,  # ← 밝은 눈 효과
        p=1.0
    ),

    # 풍경 전체를 눈으로 덮는 느낌
    "Fast-snowy-landscape": A.RandomSnow(
        brightness_coeff=3.0,  # ← 더 강한 눈덮임
        p=1.0
    )
}

# UI 시작
st.title("OCR 이미지 자동 증강기 (Albumentations 기반)")
st.write("이미지를 업로드하면, 23가지 증강 효과가 자동 적용되고 미리보기도 확인할 수 있어요.")

uploaded_files = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Start Augmentation") and uploaded_files:
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        st.subheader(f"원본 이미지 - {file.name}")
        st.image(image, use_column_width=True)

        for aug_name, augmenter in AUGMENTATIONS.items():
            augmented = augmenter(image=image_np)
            aug_image = augmented["image"]

            # 저장
            aug_folder = os.path.join(output_dir, aug_name)
            os.makedirs(aug_folder, exist_ok=True)
            filename = os.path.splitext(file.name)[0]
            out_path = os.path.join(aug_folder, f"{filename}_{aug_name}.jpg")
            Image.fromarray(aug_image).save(out_path)

            # 미리보기 표시
            st.markdown(f"**📌 {aug_name}**")
            st.image(aug_image, use_column_width=True)

    st.success("✅ 이미지 증강 완료! 'output' 폴더를 확인하세요.")
else:
    st.info("왼쪽에서 이미지를 업로드하고 'Start Augmentation' 버튼을 눌러주세요.")
