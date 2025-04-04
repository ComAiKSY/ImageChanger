import streamlit as st
import os
from PIL import Image
import numpy as np
import albumentations as A
import shutil

# 증강 목록 정의
AUGMENTATIONS = {
    "rotate": A.Rotate(limit=25, p=1.0),
    "shear": A.Affine(shear=(-16, 16), p=1.0),
    "vertical-flip": A.VerticalFlip(p=1.0),
    "horizontal-flip": A.HorizontalFlip(p=1.0),

    "crop": A.Affine(
    scale={"x": 1.3, "y": 1.0},  # 좌우로만 늘림
    translate_percent={"x": 0.0, "y": 0.0},
    fit_output=True,
    p=1.0
    ),

    "crop-and-pad": A.PadIfNeeded(min_height=256, min_width=256, border_mode=4, p=1.0),
    "perspective-transform": A.Perspective(scale=(0.05, 0.1), p=1.0),
    "elastic-transformation": A.ElasticTransform(alpha=200, sigma=25, p=1.0),

    "sharpen": A.Sharpen(
    alpha=(0.8, 1.0),         # sharpening mask 강도 최대치
    lightness=(1.5, 2.0),     # 밝기 강조 (눈부심 느낌)
    p=1.0
    ),
    "brighten": A.RandomBrightnessContrast(brightness_limit=(0.4, 0.6), contrast_limit=(0.0, 0.1), p=1.0),
    "Gamma-contrast": A.RandomGamma(
    gamma_limit=(30, 50),  # 낮은 톤 유지
    p=1.0
    ),
    "invert": A.InvertImg(p=1.0),
    "gaussian-blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    "additive-gaussian-noise": A.GaussNoise(p=1.0),

    "translate-x": A.GridDistortion(num_steps=10, distort_limit=0.2, p=1.0),
    "translate-y": A.OpticalDistortion(distort_limit=0.5, p=1.0),

    "coarse-salt": A.CoarseDropout(
    min_holes=15,        # 최소 블럭 수
    max_holes=20,        # 최대 블럭 수
    min_height=10,       # 블럭 최소 높이
    max_height=20,       # 블럭 최대 높이
    min_width=10,        # 블럭 최소 너비
    max_width=20,        # 블럭 최대 너비
    fill_value=255,      # 흰색 블럭 (소금)
    p=1.0
    ),

    "super-pixel": A.Downscale(
    scale_min=0.1, 
    scale_max=0.2, 
    p=1.0
    ),
    "emboss": A.Compose([
    A.Emboss(alpha=(0.8, 1.0), strength=(0.8, 1.0), p=1.0),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=10, p=1.0)
    ]),

    "clouds": A.Compose([
    A.RandomFog(p=1.0),  # fog_coef 제거
    A.ImageCompression(quality_lower=30, quality_upper=70, p=1.0)  # quality → quality_lower/upper
    ]),

    "fog": A.RandomFog(p=1.0),
    "snow-flakes": A.RandomSnow(brightness_coeff=2.5, p=1.0),
    "Fast-snowy-landscape": A.RandomSnow(
    brightness_coeff=3.0, 
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
