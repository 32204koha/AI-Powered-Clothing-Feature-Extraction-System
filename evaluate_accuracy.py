import csv
import json
import os
import time
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session
from transformers import pipeline

# ----------------------------------------------------
# 1. パス & 設定
# ----------------------------------------------------
BASE_DIR = r"C:\Users\Public\Documents\Kenkyu\AI-Powered-Clothing-Feature-Extraction-System"
CSV_PATH = os.path.join(BASE_DIR, "fashion-dataset", "styles.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "fashion-dataset", "images")
PALETTE_FILE = os.path.join(BASE_DIR, "color_palette.json")
CATEGORY_JSON_FILE = os.path.join(BASE_DIR, "category_prompts.json")
CATEGORY_MAP_FILE = "category_mapping.json"

# 評価する画像サンプル数の上限（全件評価する場合は None に設定）
MAX_EVAL_SAMPLES = 10

# 軽量な背景削除モデルを指定 (u2netp: 超軽量・高速)
REMBG_MODEL = "u2netp"

# ----------------------------------------------------
# 2. 外部データの読み込み
# ----------------------------------------------------
def load_color_palette():
    if not os.path.exists(PALETTE_FILE):
        return {}
    with open(PALETTE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    palette_dict = {}
    for k, v in data.items():
        display_name = v.get("label_jp", k)
        palette_dict[display_name] = {
            "bgr": v["bgr"],
            "en": v.get("en", k)
        }
    return palette_dict

def load_category_prompts():
    if not os.path.exists(CATEGORY_JSON_FILE):
        return {}
    with open(CATEGORY_JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

COLOR_PALETTE = load_color_palette()
CATEGORY_DATA = load_category_prompts()

def load_category_mapping():
    """
    細かな articleType を共通の代表グループに変換する逆引き辞書を構築する
    例: "Kurta Sets" -> "Shirts"
    """
    if not os.path.exists(CATEGORY_MAP_FILE):
        return {}
    
    with open(CATEGORY_MAP_FILE, "r", encoding="utf-8") as f:
        group_data = json.load(f)
    
    reverse_map = {}
    for group_name, variants in group_data.items():
        for v in variants:
            reverse_map[v.lower()] = group_name
    return reverse_map

CATEGORY_REVERSE_MAP = load_category_mapping()

def normalize_category(cat_name: str) -> str:
    """
    カテゴリ名をマッピング辞書を使って正規化する。
    辞書にない場合は元の名前をそのまま返す（または小文字化）。
    """
    clean_name = cat_name.strip().lower()
    return CATEGORY_REVERSE_MAP.get(clean_name, cat_name)

# CLIP用のプロンプトリスト構築
PROMPT_LIST = []
PROMPT_TO_CAT = {}
for cat_name, info in CATEGORY_DATA.items():
    prompts = info.get("prompts", [f"a photo of a {cat_name}"])
    for p in prompts:
        PROMPT_LIST.append(p)
        PROMPT_TO_CAT[p] = cat_name

# ----------------------------------------------------
# 3. AIモデルの初期化
# ----------------------------------------------------
print("==================================================")
print(" 🚀 ファッション属性抽出・正答率評価システム")
print("==================================================")
print(f"⏳ 軽量背景削除AI ({REMBG_MODEL}) をロード中...")
rembg_sess = new_session(REMBG_MODEL)

print("⏳ CLIP ゼロショット分類モデルをロード中...")
clip_classifier = pipeline(
    "zero-shot-image-classification",
    model="openai/clip-vit-base-patch32"
)
print("✨ モデルの初期化が完了しました！\n")

# ----------------------------------------------------
# 4. 解析ロジック (FastAPIコードベース)
# ----------------------------------------------------
def analyze_clothing_color(img_rgb: Image.Image) -> str:
    np_img = np.array(img_rgb)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    mask = ~((bgr_img[:, :, 0] > 240) & (bgr_img[:, :, 1] > 240) & (bgr_img[:, :, 2] > 240))
    valid_pixels = bgr_img[mask]
    if len(valid_pixels) == 0:
        return "Unknown"

    valid_hsv = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    valid_labs = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(float)

    mean_h = np.mean(valid_hsv[:, 0])
    mean_s = np.mean(valid_hsv[:, 1])
    is_monochrome = (mean_s < 30)

    all_names = list(COLOR_PALETTE.keys())
    allowed_indices = []
    
    for i, name in enumerate(all_names):
        bgr = COLOR_PALETTE[name]["bgr"]
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
    palette_labs = []
    for name in filtered_names:
        bgr_val = COLOR_PALETTE[name]["bgr"]
        lab_val = cv2.cvtColor(np.array([[bgr_val]], dtype=np.uint8), cv2.COLOR_BGR2Lab)[0][0].astype(float)
        palette_labs.append(lab_val)
    
    palette_array = np.array(palette_labs)
    distances = np.linalg.norm(valid_labs[:, np.newaxis, :] - palette_array[np.newaxis, :, :], axis=2)
    min_indices = np.argmin(distances, axis=1)

    counts = np.bincount(min_indices, minlength=len(filtered_names))
    sorted_indices = np.argsort(counts)[::-1]

    first_color = filtered_names[sorted_indices[0]] if counts[sorted_indices[0]] > 0 else "Unknown"
    
    # 🌟 カラーファミリー（基本色）を取得して返す。設定されていない場合は英語名にフォールバック
    color_info = COLOR_PALETTE.get(first_color, {})
    color_family = color_info.get("color_family")
    
    if not color_family:
        color_family = color_info.get("en", first_color)
        
    return color_family

def predict_best_category(img_rgb: Image.Image) -> str:
    try:
        predictions = clip_classifier(img_rgb, candidate_labels=PROMPT_LIST)
        if not predictions:
            return "Unknown"
        top_pred = predictions[0]
        best_prompt = top_pred["label"]
        return PROMPT_TO_CAT.get(best_prompt, "Unknown")
    except Exception:
        return "Unknown"

# ----------------------------------------------------
# 5. 評価ループの実行
# ----------------------------------------------------
def run_evaluation():
    if not os.path.exists(CSV_PATH):
        print(f"❌ エラー: CSVファイルが見つかりません ({CSV_PATH})")
        return

    eval_targets = []
    with open(CSV_PATH, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_id = row.get("id", "").strip()
            article_type = row.get("articleType", "").strip()
            base_color = row.get("baseColour", "").strip()

            img_file = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
            if os.path.exists(img_file) and article_type and base_color:
                eval_targets.append({
                    "id": img_id,
                    "file_path": img_file,
                    "gt_category": article_type,
                    "gt_color": base_color
                })

    total_samples = len(eval_targets)
    if MAX_EVAL_SAMPLES:
        eval_targets = eval_targets[:MAX_EVAL_SAMPLES]
        total_eval = len(eval_targets)
    else:
        total_eval = total_samples

    print(f"📂 検出対象画像数: {total_samples:,} 件")
    print(f"🎯 評価実行サンプル数: {total_eval:,} 件\n")
    print("--------------------------------------------------------------------------------")
    print(f"{'No.':<5} | {'ID':<8} | {'色 (推論 / 正解)':<25} | {'カテゴリー (推論 / 正解)':<35} | {'処理時間'}")
    print("--------------------------------------------------------------------------------")

    cat_correct = 0
    color_correct = 0
    total_time = 0.0

    for idx, target in enumerate(eval_targets, 1):
        start_time = time.time()

        # 画像読み込み ＆ 効率化のためのリサイズ
        raw_img = Image.open(target["file_path"]).convert("RGB")
        raw_img.thumbnail((224, 224))

        # 背景削除
        output_image_rgba = remove(raw_img, session=rembg_sess)
        bg = Image.new("RGB", output_image_rgba.size, (255, 255, 255))
        bg.paste(output_image_rgba, mask=output_image_rgba.split()[3])

        # 解析実行
        pred_color = analyze_clothing_color(bg)
        pred_cat = predict_best_category(bg)

        proc_time = (time.time() - start_time) * 1000  # ms
        total_time += proc_time

        # 一致判定（大文字小文字を無視）
        # 正規化（グループ化）を通したカテゴリ名で比較
        norm_pred_cat = normalize_category(pred_cat)
        norm_gt_cat = normalize_category(target["gt_category"])

        is_cat_match = (norm_pred_cat.lower() == norm_gt_cat.lower())
        is_color_match = (pred_color.lower() == target["gt_color"].lower())

        if is_cat_match:
            cat_correct += 1
        if is_color_match:
            color_correct += 1

        cat_str = f"{pred_cat} / {target['gt_category']}"
        color_str = f"{pred_color} / {target['gt_color']}"

        print(f"{idx:<5} | {target['id']:<8} | {color_str:<25} | {cat_str:<35} | {proc_time:.0f} ms")

    # ----------------------------------------------------
    # 6. 集計結果表示
    # ----------------------------------------------------
    avg_time = total_time / total_eval if total_eval > 0 else 0
    cat_accuracy = (cat_correct / total_eval) * 100 if total_eval > 0 else 0
    color_accuracy = (color_correct / total_eval) * 100 if total_eval > 0 else 0

    print("==================================================")
    print(" 📊 正答率・処理パフォーマンス 最終検証結果")
    print("==================================================")
    print(f" 評価サンプル数   : {total_eval:,} 件")
    print(f" ⏱️ 平均処理時間  : {avg_time:.1f} ms / 1枚")
    print(f" 🏷️ カテゴリー正答率: {cat_accuracy:.2f}% ({cat_correct}/{total_eval})")
    print(f" 🎨 カラー正答率     : {color_accuracy:.2f}% ({color_correct}/{total_eval})")
    print("==================================================")

if __name__ == "__main__":
    run_evaluation()