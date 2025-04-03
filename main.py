import streamlit as st
import os
from PIL import Image
import numpy as np
import albumentations as A
import shutil

# 증강 목록 정의 (유효한 인자만 적용된 최신 버전)
AUGMENTATIONS = {
    "rotate": A.Rotate(limit=25, p=1.0),
    "shear": A.Affine(shear=(-16, 16), p=1.0),
    "vertical-flip": A.VerticalFlip(p=1.0),
    "horizontal-flip": A.HorizontalFlip(p=1.0),
    "crop": A.RandomCrop(height=200, width=200, p=1.0),
    "crop-and-pad": A.PadIfNeeded(min_height=256, min_width=256, border_mode=0, p=1.0),
    "perspective-transform": A.Perspective(scale=(0.05, 0.1), p=1.0),
    "elastic-transformation": A.ElasticTransform(alpha=1, sigma=50, p=1.0),
    "sharpen": A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    "brighten": A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.0, p=1.0),
    "Gamma-contrast": A.RandomGamma(gamma_limit=(80, 120), p=1.0),
    "invert": A.InvertImg(p=1.0),
    "gaussian-blur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    "additive-gaussian-noise": A.GaussNoise(p=1.0),
    "translate-x": A.Affine(translate_percent={"x": 0.1}, p=1.0),
    "translate-y": A.Affine(translate_percent={"y": 0.1}, p=1.0),
    "coarse-salt": A.CoarseDropout(p=1.0),
    "super-pixel": A.Downscale(p=1.0),
    "emboss": A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
    "clouds": A.RandomFog(p=1.0),
    "fog": A.RandomFog(p=1.0),
    "snow-flakes": A.RandomSnow(brightness_coeff=2.5, p=1.0),
    "Fast-snowy-landscape": A.RandomSnow(p=1.0)
}

# UI 시작
st.title("OCR 이미지 자동 증강기 (Albumentations 기반)")
st.write("여러 이미지를 업로드하면, 23가지 증강이 자동으로 적용되어 저장됩니다.")

uploaded_files = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if st.button("Start Augmentation") and uploaded_files:
    output_dir = "output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")
        image_np = np.array(image)

        for aug_name, augmenter in AUGMENTATIONS.items():
            augmented = augmenter(image=image_np)
            aug_image = augmented["image"]

            aug_folder = os.path.join(output_dir, aug_name)
            os.makedirs(aug_folder, exist_ok=True)

            filename = os.path.splitext(file.name)[0]
            out_path = os.path.join(aug_folder, f"{filename}_{aug_name}.jpg")
            Image.fromarray(aug_image).save(out_path)

    st.success("✅ 이미지 증강 완료! 'output' 폴더를 확인하세요.")
else:
    st.info("왼쪽에서 이미지를 업로드하고 'Start Augmentation' 버튼을 눌러주세요.")
