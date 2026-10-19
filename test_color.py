from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io
import os
import json
import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "test_outputs"
os.makedirs(TEMP_DIR, exist_ok=True)

PALETTE_FILE = "color_palette.json"

# プロンプト生成用のテンプレート定義（モデル検証・属性アライメント用）
PROMPT_TEMPLATES = [
    "a photo of a casual {color} clothing item",
    "a close-up photo of a {color} apparel garment",
    "a photo of a {color} fashion product for men or women"
]

def generate_color_prompts(color_label: str) -> list[str]:
    """
    指定された色ラベルに基づいて洗練されたプロンプト選択肢（バリエーション）を生成する
    """
    return [template.format(color=color_label.lower()) for template in PROMPT_TEMPLATES]

def load_color_palette():
    if not os.path.exists(PALETTE_FILE):
        print(f"⚠️ 警告: {PALETTE_FILE} が見つかりません。デフォルト値を使用します。")
        return {
            "ホワイト": {"bgr": [245, 245, 245], "en": "white"},
            "ブラック": {"bgr": [20, 20, 20], "en": "black"},
            "グレー": {"bgr": [128, 128, 128], "en": "grey"},
            "ネイビー": {"bgr": [120, 50, 20], "en": "navy blue"},
            "ブルー": {"bgr": [200, 100, 40], "en": "blue"},
            "水色": {"bgr": [230, 210, 150], "en": "light blue"},
            "レッド": {"bgr": [40, 40, 200], "en": "red"},
            "グリーン": {"bgr": [40, 160, 60], "en": "green"},
            "オリーブ": {"bgr": [45, 95, 95], "en": "olive"},
            "ブラウン": {"bgr": [35, 65, 115], "en": "brown"}
        }
    
    with open(PALETTE_FILE, "r", encoding="utf-8") as f:
        palette_data = json.load(f)
    
    palette_dict = {}
    for color_key, val in palette_data.items():
        display_name = val.get("label_jp", color_key)
        palette_dict[display_name] = {
            "bgr": val["bgr"],
            "en": val.get("en", color_key)
        }
        
    return palette_dict

COLOR_PALETTE = load_color_palette()
print(f"🎨 カラーパレット読み込み完了: {list(COLOR_PALETTE.keys())}")

# AIモデル常駐
sessions = {}
TARGET_MODELS = ["u2net", "u2netp", "isnet-general-use", "bria-rmbg"]
for name in TARGET_MODELS:
    try:
        sessions[name] = new_session(name)
        print(f"  - AIモデル常駐成功: {name}")
    except Exception as e:
        print(f"  - AIモデル常駐スキップ ({name}): {e}")

def analyze_clothing_color(img_rgb: Image.Image) -> dict:
    """
    HSV色相大分類フィルター + Lab色空間による詳細多数決判定
    """
    np_img = np.array(img_rgb)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    mask = ~((bgr_img[:, :, 0] > 240) & (bgr_img[:, :, 1] > 240) & (bgr_img[:, :, 2] > 240))
    valid_pixels = bgr_img[mask]
    if len(valid_pixels) == 0:
        return {
            "first": "不明",
            "second": "-",
            "first_prompts": [],
            "second_prompts": []
        }

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

    first_color = filtered_names[sorted_indices[0]] if counts[sorted_indices[0]] > 0 else "不明"
    second_color = "-"
    if len(sorted_indices) > 1 and counts[sorted_indices[1]] > 0:
        second_color = filtered_names[sorted_indices[1]]

    # 判定結果に応じた洗練されたプロンプト生成
    first_en = COLOR_PALETTE.get(first_color, {}).get("en", first_color)
    first_prompts = generate_color_prompts(first_en) if first_color != "不明" else []

    second_prompts = []
    if second_color != "-":
        second_en = COLOR_PALETTE.get(second_color, {}).get("en", second_color)
        second_prompts = generate_color_prompts(second_en)

    return {
        "first": first_color,
        "second": second_color,
        "first_prompts": first_prompts,
        "second_prompts": second_prompts
    }

@app.post("/test-remove-bg")
async def test_remove_bg(
    file: UploadFile = File(...),
    model_name: str = Form("u2net")
):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        session = sessions.get(model_name, sessions.get("u2net"))
        output_image_rgba = remove(input_image, session=session)
        
        bg = Image.new("RGB", output_image_rgba.size, (255, 255, 255))
        bg.paste(output_image_rgba, mask=output_image_rgba.split()[3])
        
        color_result = analyze_clothing_color(bg)

        final_path = os.path.join(TEMP_DIR, f"result_{model_name}.jpg")
        bg.save(final_path, "JPEG")

        return {
            "status": "success",
            "model": model_name,
            "detected_color": color_result["first"],
            "second_color": color_result["second"],
            "generated_prompts": {
                "first_color_prompts": color_result["first_prompts"],
                "second_color_prompts": color_result["second_prompts"]
            },
            "image_url": f"http://localhost:8000/test-image/{model_name}"
        }
    except Exception as e:
        print(f"エラー: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/test-image/{model_name}")
async def get_test_image(model_name: str):
    path = os.path.join(TEMP_DIR, f"result_{model_name}.jpg")
    if os.path.exists(path):
        return FileResponse(path)
    return FileResponse(os.path.join(TEMP_DIR, "result_u2net.jpg"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)