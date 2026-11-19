from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, List, Optional
import io
import os
import json
import uuid
import numpy as np
import cv2
import torch
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

app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

PALETTE_FILE = "color_palette.json"
CATEGORY_JSON_FILE = "category_prompts.json"

# =========================================================
# 日本語グループ統合マッピング定義（正しさ・納得感重視）
# =========================================================
COLOR_GROUP_MAP = {
    # ブラック系
    "black": "ブラック系", "charcoal": "ブラック系", "darkslategray": "ブラック系", "dimgray": "ブラック系",
    "ブラック": "ブラック系", "黒": "ブラック系", "チャコール": "ブラック系",
    # ホワイト系
    "white": "ホワイト系", "aliceblue": "ホワイト系", "antiquewhite": "ホワイト系", "linen": "ホワイト系", "seashell": "ホワイト系",
    "ホワイト": "ホワイト系", "白": "ホワイト系", "オフホワイト": "ホワイト系",
    # グレー系
    "gray": "グレー系", "grey": "グレー系", "darkgray": "グレー系", "lightgray": "グレー系",
    "silver": "グレー系", "gainsboro": "グレー系", "slategray": "グレー系", "lightslategray": "グレー系",
    "グレー": "グレー系", "シルバー": "グレー系",
    # ネイビー系
    "navy blue": "ネイビー系", "navy": "ネイビー系", "midnightblue": "ネイビー系", "darkslateblue": "ネイビー系",
    "ネイビー": "ネイビー系", "紺": "ネイビー系",
    # ブルー系
    "blue": "ブルー系", "steelblue": "ブルー系", "lightsteelblue": "ブルー系", "deepskyblue": "ブルー系",
    "dodgerblue": "ブルー系", "powderblue": "ブルー系", "lightblue": "ブルー系", "cadetblue": "ブルー系",
    "ブルー": "ブルー系", "青": "ブルー系", "水色": "ブルー系",
    # ブラウン・ベージュ系
    "brown": "ブラウン・ベージュ系", "sienna": "ブラウン・ベージュ系", "beige": "ブラウン・ベージュ系",
    "khaki": "ブラウン・ベージュ系", "darkkhaki": "ブラウン・ベージュ系", "wheat": "ブラウン・ベージュ系",
    "burlywood": "ブラウン・ベージュ系", "palegoldenrod": "ブラウン・ベージュ系", "sandybrown": "ブラウン・ベージュ系",
    "navajowhite": "ブラウン・ベージュ系", "rosybrown": "ブラウン・ベージュ系",
    "ブラウン": "ブラウン・ベージュ系", "茶": "ブラウン・ベージュ系", "ベージュ": "ブラウン・ベージュ系", "カーキ": "ブラウン・ベージュ系",
    # グリーン系
    "green": "グリーン系", "darkseagreen": "グリーン系", "seagreen": "グリーン系", "darkolivegreen": "グリーン系",
    "mediumaquamarine": "グリーン系", "yellowgreen": "グリーン系", "darkcyan": "グリーン系",
    "グリーン": "グリーン系", "緑": "グリーン系", "オリーブ": "グリーン系",
    # レッド・ピンク系
    "red": "レッド・ピンク系", "crimson": "レッド・ピンク系", "firebrick": "レッド・ピンク系",
    "indianred": "レッド・ピンク系", "tomato": "レッド・ピンク系", "orangered": "レッド・ピンク系",
    "hotpink": "レッド・ピンク系", "mistyrose": "レッド・ピンク系", "pink": "レッド・ピンク系",
    "レッド": "レッド・ピンク系", "赤": "レッド・ピンク系", "ピンク": "レッド・ピンク系",
    # イエロー・オレンジ系
    "yellow": "イエロー・オレンジ系", "goldenrod": "イエロー・オレンジ系", "orange": "イエロー・オレンジ系",
    "イエロー": "イエロー・オレンジ系", "黄": "イエロー・オレンジ系", "オレンジ": "イエロー・オレンジ系",
    # パープル系
    "purple": "パープル系", "mediumpurple": "パープル系", "mediumorchid": "パープル系",
    "plum": "パープル系", "slateblue": "パープル系",
    "パープル": "パープル系", "紫": "パープル系",
}

CATEGORY_GROUP_MAP = {
    # 1. Tシャツ・カットソー
    "tshirts": "Tシャツ・カットソー",
    "lounge tshirts": "Tシャツ・カットソー",
    "tops": "Tシャツ・カットソー",
    "camisoles": "Tシャツ・カットソー",
    "innerwear vests": "Tシャツ・カットソー",
    # 2. シャツ・ブラウス
    "shirts": "シャツ・ブラウス",
    "tunics": "シャツ・ブラウス",
    "kurtis": "シャツ・ブラウス",
    "kurtas": "シャツ・ブラウス",
    # 3. スウェット・ニット
    "sweatshirts": "スウェット・ニット",
    "sweaters": "スウェット・ニット",
    # 4. ジャケット・アウター
    "blazers": "ジャケット・アウター",
    "nehru jackets": "ジャケット・アウター",
    "jackets": "ジャケット・アウター",
    "coats": "ジャケット・アウター",
    # 5. パンツ・ボトムス
    "jeans": "パンツ・ボトムス",
    "trousers": "パンツ・ボトムス",
    "jeggings": "パンツ・ボトムス",
    "capris": "パンツ・ボトムス",
    "lounge pants": "パンツ・ボトムス",
    "rain trousers": "パンツ・ボトムス",
    "patiala": "パンツ・ボトムス",
    # 6. ショートパンツ
    "shorts": "ショートパンツ",
    "lounge shorts": "ショートパンツ",
    # 7. スカート
    "skirts": "スカート",
    # 8. ワンピース・セットアップ
    "dresses": "ワンピース・セットアップ",
    "kurta sets": "ワンピース・セットアップ",
    "salwar and dupatta": "ワンピース・セットアップ",
    "rompers": "ワンピース・セットアップ",
    "nightdress": "ワンピース・セットアップ",
    "baby dolls": "ワンピース・セットアップ",
    # 9. シューズ・バッグ・小物
    "sports shoes": "シューズ・バッグ・小物",
    "casual shoes": "シューズ・バッグ・小物",
    "formal shoes": "シューズ・バッグ・小物",
    "sandals": "シューズ・バッグ・小物",
    "sports sandals": "シューズ・バッグ・小物",
    "flip flops": "シューズ・バッグ・小物",
    "flats": "シューズ・バッグ・小物",
    "heels": "シューズ・バッグ・小物",
    "handbags": "シューズ・バッグ・小物",
    "messenger bag": "シューズ・バッグ・小物",
    "duffel bag": "シューズ・バッグ・小物",
    "laptop bag": "シューズ・バッグ・小物",
    "rucksacks": "シューズ・バッグ・小物",
    "backpacks": "シューズ・バッグ・小物",
    "watches": "シューズ・バッグ・小物",
    "belts": "シューズ・バッグ・小物",
    "bracelet": "シューズ・バッグ・小物",
    "hat": "シューズ・バッグ・小物",
    "caps": "シューズ・バッグ・小物",
}


def map_to_color_group(raw_color: str) -> str:
    if not raw_color or raw_color == "不明":
        return "不明"
    if raw_color.endswith("系"):
        return raw_color
    key = raw_color.strip().lower()
    if key in COLOR_GROUP_MAP:
        return COLOR_GROUP_MAP[key]
    for k, group in COLOR_GROUP_MAP.items():
        if k in key:
            return group
    return "その他・不明"


def map_to_category_group(raw_category: str) -> str:
    if not raw_category or raw_category in ["不明", "エラー"]:
        return "シューズ・バッグ・小物"
    # すでに日本語グループ名の場合
    japanese_groups = {
        "Tシャツ・カットソー", "シャツ・ブラウス", "スウェット・ニット",
        "ジャケット・アウター", "パンツ・ボトムス", "ショートパンツ",
        "スカート", "ワンピース・セットアップ", "シューズ・バッグ・小物"
    }
    if raw_category in japanese_groups:
        return raw_category

    key = raw_category.strip().lower()
    if key in CATEGORY_GROUP_MAP:
        return CATEGORY_GROUP_MAP[key]

    # 部分一致フォールバック
    if "tshirt" in key or "top" in key:
        return "Tシャツ・カットソー"
    if "shirt" in key or "tunic" in key or "kurt" in key:
        return "シャツ・ブラウス"
    if "sweat" in key or "sweater" in key or "knit" in key:
        return "スウェット・ニット"
    if "jacket" in key or "blazer" in key or "coat" in key:
        return "ジャケット・アウター"
    if "short" in key:
        return "ショートパンツ"
    if "pant" in key or "jean" in key or "trouser" in key:
        return "パンツ・ボトムス"
    if "skirt" in key:
        return "スカート"
    if "dress" in key or "set" in key:
        return "ワンピース・セットアップ"

    return "シューズ・バッグ・小物"


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
        items = json.load(f)

    # 既存データの日本語グループ表記への互換保証
    for item in items:
        item["color"] = map_to_color_group(item.get("color", "不明"))
        item["category"] = map_to_category_group(item.get("category", "不明"))
    return items


def save_wardrobe(items: list) -> None:
    with open(WARDROBE_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


# =========================================================
# 画像解析関数（色・カテゴリー抽出・CLIP特徴ベクトル類似度検索）
# =========================================================
def analyze_clothing_color(img_rgb: Image.Image) -> str:
    """
    HSV色相大分類フィルター + Lab色空間による最頻色判定後、日本語10色グループへ統合
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
    is_monochrome = mean_s < 25

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
    raw_best_color = filtered_names[best_idx] if counts[best_idx] > 0 else "不明"

    return map_to_color_group(raw_best_color)


def predict_best_category(img_rgb: Image.Image) -> dict:
    """
    CLIPを用いて最もスコアが高いカテゴリーを抽出し、日本語9カテゴリーグループへ統合
    """
    try:
        predictions = clip_classifier(img_rgb, candidate_labels=PROMPT_LIST)
        if not predictions:
            return {"category": "シューズ・バッグ・小物", "score": 0.0}

        top_pred = predictions[0]
        best_prompt = top_pred["label"]
        score = round(float(top_pred["score"]), 4)
        raw_category = PROMPT_TO_CAT.get(best_prompt, "不明")
        grouped_category = map_to_category_group(raw_category)

        return {
            "category": grouped_category,
            "score": score,
        }
    except Exception as e:
        print(f"CLIP推論エラー: {e}")
        return {"category": "シューズ・バッグ・小物", "score": 0.0}


def extract_image_embedding(img_rgb: Image.Image) -> List[float]:
    """
    常駐済みのCLIP画像エンコーダから正規化済み特徴ベクトル（512次元）を抽出する
    """
    try:
        inputs = clip_classifier.image_processor(images=img_rgb, return_tensors="pt")
        with torch.no_grad():
            features = clip_classifier.model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)
        return features[0].cpu().numpy().tolist()
    except Exception as e:
        print(f"特徴ベクトル抽出エラー: {e}")
        return []


def find_most_similar_item(embedding: List[float], wardrobe: List[dict]) -> Optional[dict]:
    """
    登録済みクローゼットから最も視覚的類似度が高いアイテムを検索する（しきい値: 0.85以上）
    """
    if not embedding or not wardrobe:
        return None

    target_vec = np.array(embedding, dtype=np.float32)
    best_sim = -1.0
    best_item = None
    best_index = -1

    for idx, item in enumerate(wardrobe):
        item_emb = item.get("embedding")
        if not item_emb:
            continue
        cand_vec = np.array(item_emb, dtype=np.float32)
        sim = float(np.dot(target_vec, cand_vec))
        if sim > best_sim:
            best_sim = sim
            best_item = item
            best_index = idx + 1

    # 類似度 85% (0.85) 以上を類似候補として検出
    if best_item is not None and best_sim >= 0.85:
        return {
            "item_id": best_item["id"],
            "index": best_index,
            "image_url": best_item["image_url"],
            "category": best_item["category"],
            "color": best_item["color"],
            "similarity": round(best_sim, 4),
            "similarity_percent": round(best_sim * 100, 1),
            "is_likely_identical": best_sim >= 0.93,
        }
    return None


# =========================================================
# API エンドポイント
# =========================================================
class RegisterItemRequest(BaseModel):
    item_id: str
    image_url: str
    color: str
    category: str
    answers: Dict[str, str]
    embedding: List[float] = []
    has_similar_candidate: bool = False
    is_same_item: bool = False
    same_as_item_id: Optional[str] = None


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    ステップ1: 背景削除、日本語グループ化された色・カテゴリー抽出、類似服検索、重複先読みを行う
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

        # 色・カテゴリー・CLIP特徴ベクトルの抽出
        detected_color = analyze_clothing_color(bg)
        cat_result = predict_best_category(bg)
        detected_category = cat_result["category"]
        embedding = extract_image_embedding(bg)

        current_wardrobe = load_wardrobe()

        # 既登録アイテムとの視覚的類似度チェック
        similar_item = find_most_similar_item(embedding, current_wardrobe)

        # 「別の服」として登録した場合の重複状況（3着以上になるか）
        virtual_wardrobe_diff = current_wardrobe + [{
            "color": detected_color,
            "category": detected_category,
            "is_same_item": False,
        }]
        scorer_diff = DeclutterScorer(virtual_wardrobe_diff)
        dup_status_if_different = scorer_diff.check_duplicates(
            detected_color, detected_category, is_same_item=False
        )

        # 「同じ服」として登録した場合の重複状況（重複カウント除外）
        dup_status_if_same = {
            "has_duplicate_color": False,
            "has_duplicate_category": False,
        }

        return {
            "status": "success",
            "item_id": item_id,
            "image_url": f"/images/{filename}",
            "color": detected_color,
            "category": detected_category,
            "confidence": cat_result["score"],
            "embedding": embedding,
            "similar_item": similar_item,
            "dup_status": dup_status_if_different,
            "dup_status_if_different": dup_status_if_different,
            "dup_status_if_same": dup_status_if_same,
        }
    except Exception as e:
        print(f"画像解析エラー: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.post("/register-item")
async def register_item(req: RegisterItemRequest):
    """
    ステップ2: 質問回答と同一服判定を受け取り、保存した上で全アイテムのスコアを再計算する
    """
    try:
        wardrobe = load_wardrobe()

        new_item = {
            "id": req.item_id,
            "image_url": req.image_url,
            "color": map_to_color_group(req.color),
            "category": map_to_category_group(req.category),
            "answers": req.answers,
            "embedding": req.embedding,
            "has_similar_candidate": req.has_similar_candidate,
            "is_same_item": req.is_same_item,
            "same_as_item_id": req.same_as_item_id,
        }
        wardrobe.append(new_item)

        # 全アイテムのスコアを一括再計算
        updated_wardrobe = DeclutterScorer.recalculate_all(wardrobe)
        save_wardrobe(updated_wardrobe)

        registered_item = next(item for item in updated_wardrobe if item["id"] == req.item_id)
        # レスポンスでは巨大なembedding配列を除外して返す
        clean_registered = {k: v for k, v in registered_item.items() if k != "embedding"}
        summary = DeclutterScorer.summarize_wardrobe(updated_wardrobe)

        return {
            "status": "success",
            "item": clean_registered,
            "summary": summary,
        }
    except Exception as e:
        print(f"登録エラー: {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/diagnose")
async def diagnose_wardrobe():
    """
    クローゼット全体の集計診断
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
    クローゼットの登録一覧（通信軽量化のためembeddingは除外して返す）
    """
    wardrobe = load_wardrobe()
    updated_wardrobe = DeclutterScorer.recalculate_all(wardrobe)
    save_wardrobe(updated_wardrobe)

    clean_items = [
        {k: v for k, v in item.items() if k != "embedding"}
        for item in updated_wardrobe
    ]

    return {
        "status": "success",
        "total_count": len(clean_items),
        "closet_items": clean_items,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)