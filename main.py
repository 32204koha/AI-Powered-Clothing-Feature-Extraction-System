from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict
import io
import os
import json
import uuid
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
from transformers import pipeline

from scorer import DeclutterScorer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# データ保存用ディレクトリとJSONファイルの定義
IMAGES_DIR = "saved_images"
WARDROBE_FILE = "wardrobe.json"
os.makedirs(IMAGES_DIR, exist_ok=True)

# 保存した切り抜き画像をURL (/images/xxx.jpg) で配信できるようにマウント
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

PALETTE_FILE = "color_palette.json"
CATEGORY_JSON_FILE = "category_prompts.json"


def load_color_palette() -> dict:
    if not os.path.exists(PALETTE_FILE):
        print(f"⚠️ 警告: {PALETTE_FILE} が見つかりません。デフォルト値を使用します。")
        return {
            "ホワイト": [245, 245, 245],
            "ブラック": [20, 20, 20],
            "グレー": [128, 128, 128],
            "ネイビー": [120, 50, 20],
            "ブルー": [200, 100, 40],
            "水色": [230, 210, 150],
            "レッド": [40, 40, 200],
            "グリーン": [40, 160, 60],
            "オリーブ": [45, 95, 95],
            "ブラウン": [35, 65, 115],
        }

    with open(PALETTE_FILE, "r", encoding="utf-8") as f:
        palette_data = json.load(f)

    palette_dict = {}
    for color_key, val in palette_data.items():
        display_name = val.get("label_jp", color_key)
        palette_dict[display_name] = val["bgr"]

    return palette_dict


def load_category_prompts() -> dict:
    if not os.path.exists(CATEGORY_JSON_FILE):
        print(f"⚠️ 警告: {CATEGORY_JSON_FILE} が見つかりません。デフォルト値を使用します。")
        return {
            "Shirts": {
                "prompts": [
                    "a photo of a casual button-up shirt for men",
                    "a collared dress shirt",
                ]
            },
            "Tshirts": {
                "prompts": [
                    "a photo of a plain casual t-shirt",
                    "a crewneck cotton t-shirt",
                ]
            },
        }

    with open(CATEGORY_JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


COLOR_PALETTE = load_color_palette()
print("🎨 カラーパレット読み込み完了")

CATEGORY_DATA = load_category_prompts()
PROMPT_LIST = []
PROMPT_TO_CAT = {}

for cat_name, info in CATEGORY_DATA.items():
    prompts = info.get("prompts", [f"a photo of a {cat_name}"])
    for p in prompts:
        PROMPT_LIST.append(p)
        PROMPT_TO_CAT[p] = cat_name

print("👕 カテゴリープロンプト読み込み完了")

# 背景削除モデルを u2net 1つに固定して常駐
print("⏳ 背景削除モデル (u2net) を読み込み中...")
u2net_session = new_session("u2net")
print("✨ u2net のロードが完了しました！")

print("⏳ CLIP ゼロショット画像分類モデルを読み込み中...")
clip_classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32",
)
print("✨ CLIPモデルのロードが完了しました！")


# =========================================================
# クローゼットデータ（wardrobe.json）読み書きヘルパー
# =========================================================
def load_wardrobe() -> list:
    if not os.path.exists(WARDROBE_FILE):
        return []
    with open(WARDROBE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_wardrobe(items: list) -> None:
    with open(WARDROBE_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# =========================================================
# 画像解析関数（色・カテゴリー抽出）
# =========================================================
def analyze_clothing_color(img_rgb: Image.Image) -> str:
    """
    HSV色相大分類フィルター + Lab色空間による最頻色判定
    """
    np_img = np.array(img_rgb)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    mask = ~((bgr_img[:, :, 0] > 240) & (bgr_img[:, :, 1] > 240) & (bgr_img[:, :, 2] > 240))
    valid_pixels = bgr_img[mask]
    if len(valid_pixels) == 0:
        return "不明"

    valid_hsv = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    valid_labs = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(float)

    mean_h = np.mean(valid_hsv[:, 0])
    mean_s = np.mean(valid_hsv[:, 1])
    is_monochrome = mean_s < 30

    all_names = list(COLOR_PALETTE.keys())
    allowed_indices = []

    for i, name in enumerate(all_names):
        bgr = COLOR_PALETTE[name]
        p_hsv = cv2.cvtColor(np.array([[bgr]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0][0]
        p_h, p_s = p_hsv[0], p_hsv[1]

        if is_monochrome:
            if p_s < 50 or any(k in name for k in ["白", "黒", "グレー", "White", "Black", "Grey"]):
                allowed_indices.append(i)
        else:
            h_diff = min(abs(mean_h - p_h), 180 - abs(mean_h - p_h))
            if h_diff <= 35:
                allowed_indices.append(i)

    if len(allowed_indices) == 0:
        allowed_indices = list(range(len(all_names)))

    filtered_names = [all_names[idx] for idx in allowed_indices]
    palette_labs = [
        cv2.cvtColor(np.array([[COLOR_PALETTE[name]]], dtype=np.uint8), cv2.COLOR_BGR2Lab)[0][0].astype(float)
        for name in filtered_names
    ]
    palette_array = np.array(palette_labs)

    distances = np.linalg.norm(valid_labs[:, np.newaxis, :] - palette_array[np.newaxis, :, :], axis=2)
    min_indices = np.argmin(distances, axis=1)

    counts = np.bincount(min_indices, minlength=len(filtered_names))
    best_idx = int(np.argmax(counts))

    return filtered_names[best_idx] if counts[best_idx] > 0 else "不明"


def predict_best_category(img_rgb: Image.Image) -> dict:
    """
    CLIPを用いて最もスコアが高いカテゴリーを1つ選択
    """
    try:
        predictions = clip_classifier(img_rgb, candidate_labels=PROMPT_LIST)
        if not predictions:
            return {"category": "不明", "score": 0.0}

        top_pred = predictions[0]
        best_prompt = top_pred["label"]
        score = round(float(top_pred["score"]), 4)
        best_category = PROMPT_TO_CAT.get(best_prompt, "不明")

        return {
            "category": best_category,
            "score": score,
        }
    except Exception as e:
        print(f"CLIP推論エラー: {e}")
        return {"category": "エラー", "score": 0.0}


# =========================================================
# API エンドポイント
# =========================================================
class RegisterItemRequest(BaseModel):
    item_id: str
    image_url: str
    color: str
    category: str
    answers: Dict[str, str]


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    ステップ1: 画像の背景を削除して固有IDで保存し、色・カテゴリーおよび登録後の重複状況を返す
    """
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")

        # u2net で背景削除＆白背景合成
        output_image_rgba = remove(input_image, session=u2net_session)
        bg = Image.new("RGB", output_image_rgba.size, (255, 255, 255))
        bg.paste(output_image_rgba, mask=output_image_rgba.split()[3])

        # 固有IDで切り抜き画像を保存
        item_id = f"item_{uuid.uuid4().hex[:8]}"
        filename = f"{item_id}.jpg"
        save_path = os.path.join(IMAGES_DIR, filename)
        bg.save(save_path, "JPEG")

        # 色とカテゴリーの抽出
        detected_color = analyze_clothing_color(bg)
        cat_result = predict_best_category(bg)
        detected_category = cat_result["category"]

        # この服を登録した場合の重複状況（3着以上になるか）を先読み判定
        current_wardrobe = load_wardrobe()
        virtual_wardrobe = current_wardrobe + [{"color": detected_color, "category": detected_category}]
        temp_scorer = DeclutterScorer(virtual_wardrobe)
        dup_status = temp_scorer.check_duplicates(detected_color, detected_category)

        return {
            "status": "success",
            "item_id": item_id,
            "image_url": f"/images/{filename}",
            "color": detected_color,
            "category": detected_category,
            "confidence": cat_result["score"],
            "dup_status": dup_status,
        }
    except Exception as e:
        print(f"画像解析エラー: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/register-item")
async def register_item(req: RegisterItemRequest):
    """
    ステップ2: 質問回答を受け取り、クローゼットに保存した上で全アイテムのスコアを再計算する
    """
    try:
        wardrobe = load_wardrobe()

        new_item = {
            "id": req.item_id,
            "image_url": req.image_url,
            "color": req.color,
            "category": req.category,
            "answers": req.answers,
        }
        wardrobe.append(new_item)

        # 登録順番による優劣をなくすため、全アイテムのスコアを一括再計算
        updated_wardrobe = DeclutterScorer.recalculate_all(wardrobe)
        save_wardrobe(updated_wardrobe)

        # 今回登録したアイテムの最新計算結果を取得
        registered_item = next(item for item in updated_wardrobe if item["id"] == req.item_id)
        summary = DeclutterScorer.summarize_wardrobe(updated_wardrobe)

        return {
            "status": "success",
            "item": registered_item,
            "summary": summary,
        }
    except Exception as e:
        print(f"登録エラー: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/diagnose")
async def diagnose_wardrobe():
    """
    クローゼット全体の集計診断（再計算後の平均スコアと判定別内訳を返す）
    """
    wardrobe = load_wardrobe()
    updated_wardrobe = DeclutterScorer.recalculate_all(wardrobe)
    save_wardrobe(updated_wardrobe)

    summary = DeclutterScorer.summarize_wardrobe(updated_wardrobe)
    return {
        "status": "success",
        "summary": summary,
    }


@app.get("/closet")
async def get_closet():
    """
    クローゼットの登録一覧（最新の重複状況で再計算した結果を返す）
    """
    wardrobe = load_wardrobe()
    updated_wardrobe = DeclutterScorer.recalculate_all(wardrobe)
    save_wardrobe(updated_wardrobe)

    return {
        "status": "success",
        "total_count": len(updated_wardrobe),
        "closet_items": updated_wardrobe,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)