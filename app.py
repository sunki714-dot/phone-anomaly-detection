"""Gradio demo: upload a phone-screen photo -> verdict, defect type, grade, price.

Usage:
    python app.py                      # fit on startup (downloads backbone weights)
    python app.py --memory-bank artifacts/memory_bank.pt   # load a saved bank
"""

import argparse
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phone_anomaly.config import load_config, resolve_device
from phone_anomaly.data import prepare_dataset
from phone_anomaly.features import FeatureExtractor
from phone_anomaly.models import PatchCore
from phone_anomaly.postprocess import (
    DEFECT_GRADE,
    DEFECT_TYPES,
    classify_defect,
    detect_stuck_pixel,
    phone_mask,
)
from phone_anomaly.pricing import MODEL_BASE_PRICE, estimate_price, price_range


def build_detector(cfg, device, memory_bank_path=None):
    """Load a saved memory bank if given, otherwise fit fresh from data/demo."""
    extractor = FeatureExtractor(
        backbone=cfg["backbone"]["name"],
        layers=cfg["backbone"]["layers"],
        img_size=cfg["img_size"],
        device=device,
    )
    if memory_bank_path and os.path.exists(memory_bank_path):
        print(f"Loading memory bank from {memory_bank_path}")
        return PatchCore.load(memory_bank_path, extractor)

    print("Fitting PatchCore on startup...")
    data = prepare_dataset(
        cfg["data"]["good_dir"],
        cfg["data"]["defect_dir"],
        train_ratio=cfg["data"]["train_ratio"],
        use_demo_if_missing=cfg["data"]["use_demo_if_missing"],
        seed=cfg["seed"],
    )
    return PatchCore(
        extractor,
        coreset_ratio=cfg["patchcore"]["coreset_ratio"],
        n_projection=cfg["patchcore"]["n_projection"],
        smoothing_sigma=cfg["patchcore"]["smoothing_sigma"],
        seed=cfg["seed"],
    ).fit(data["train_good"])


def _verdict_card(color, title, subtitle, footer):
    return (
        f"<div style='text-align:center;padding:22px;background:{color};"
        "border-radius:12px;color:white'>"
        f"<div style='font-size:34px;font-weight:bold'>{title}</div>"
        f"<div style='font-size:22px;margin-top:6px'>{subtitle}</div>"
        f"<div style='margin-top:6px'>{footer}</div></div>"
    )


def make_diagnose(detector, img_size, threshold):
    """Return a Gradio callback closed over the fitted detector."""

    def diagnose(image, model_name, years_old):
        if image is None:
            return None, "이미지를 올려주세요", "", ""

        amap, score = detector.anomaly_map(image, mask_fn=phone_mask)
        is_defect = score >= threshold
        stuck, loc, area = detect_stuck_pixel(image)

        # overlay the (masked) heatmap on the resized image
        img = np.array(image.convert("RGB").resize((img_size, img_size)))
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        amap_show = np.ma.masked_where(amap <= 1e-9, amap)  # background transparent
        ax.imshow(amap_show, cmap="jet", alpha=0.45, vmin=0, vmax=threshold)
        if stuck and not is_defect:
            cx = loc[0] * img_size / image.width
            cy = loc[1] * img_size / image.height
            ax.add_patch(plt.Circle((cx, cy), 12, color="lime", fill=False, lw=2))
        ax.axis("off")
        out = "_result.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)

        if is_defect:
            key, feat = classify_defect(amap)
            kind = DEFECT_TYPES[key]
            grade = DEFECT_GRADE[key]
            verdict = _verdict_card(
                "#EF4444", "⚠️ 비정상", kind, f"이상점수 {score:.2f} (임계값 {threshold:.2f})"
            )
            detail = (
                f"### 진단 결과\n- **유형**: {kind}\n- **등급**: {grade}등급\n"
                f"- **이상점수**: {score:.2f}\n"
                f"- 판정 근거: area={feat.get('area')}, spread={feat.get('spread')}, "
                f"aniso={feat.get('aniso')}"
            )
        elif stuck:
            grade = "B"
            verdict = _verdict_card(
                "#F59E0B", "⚠️ 비정상", "점결함(데드/스턱 픽셀)",
                f"이상점수 {score:.2f} · 픽셀검출 ✓",
            )
            detail = (
                f"### 진단 결과\n- **유형**: 점결함(데드/스턱 픽셀)\n- **등급**: {grade}등급\n"
                f"- **검출 위치**: {loc} (면적 {area}px)\n"
                "- 전체 이상점수는 낮지만 화면 영역에서 색을 띤 작은 점이 검출됨."
            )
        else:
            grade = "S"
            verdict = _verdict_card(
                "#10B981", "✅ 정상", "", f"이상점수 {score:.2f} (임계값 {threshold:.2f})"
            )
            detail = f"### 진단 결과\n- **등급**: {grade}등급\n정상 범위입니다. (히트맵 전반이 차가운 색)"

        price = estimate_price(model_name, years_old, grade)
        p_lo, p_hi = price_range(price)
        price_md = (
            f"## 💰 예상 시세\n# {price}만원\n범위: {p_lo} ~ {p_hi}만원\n\n"
            f"*{model_name} · {years_old}년차 · {grade}등급 기준*"
        )
        return out, verdict, detail, price_md

    return diagnose


def build_ui(diagnose):
    with gr.Blocks(title="중고폰 결함 진단 + 시세 산정") as demo:
        gr.Markdown(
            "# 📱 중고폰 결함 진단기\n"
            "사진을 올리면 **정상/비정상 판정 + 결함 유형 + 등급 + 예상 시세**를 알려줍니다."
        )
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="폰 화면 사진 업로드")
                model_dd = gr.Dropdown(
                    choices=list(MODEL_BASE_PRICE.keys()), value="iPhone 13", label="모델 선택"
                )
                years_sl = gr.Slider(0, 5, value=2, step=1, label="출시 후 경과 년수")
                btn = gr.Button("🔍 진단하기", variant="primary")
            with gr.Column():
                out_img = gr.Image(label="이상 히트맵")
                out_verdict = gr.HTML()
                out_price = gr.Markdown()
                out_detail = gr.Markdown()
        btn.click(
            diagnose,
            inputs=[inp, model_dd, years_sl],
            outputs=[out_img, out_verdict, out_detail, out_price],
        )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Launch the defect-diagnosis demo.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--memory-bank", default=None, help="path to a saved bank (optional)")
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg["device"])
    print(f"Device: {device}")

    detector = build_detector(cfg, device, args.memory_bank)
    threshold = cfg["patchcore"]["threshold"]
    diagnose = make_diagnose(detector, cfg["img_size"], threshold)
    demo = build_ui(diagnose)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
