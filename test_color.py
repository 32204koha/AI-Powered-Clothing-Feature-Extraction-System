from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io
import os
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

# 起動時にモデルを常駐
sessions = {}
TARGET_MODELS = ["u2net", "u2netp", "isnet-general-use", "bria-rmbg"]
for name in TARGET_MODELS:
    try:
        sessions[name] = new_session(name)
    except:
        pass

# 🌟 判定したい基本カラーパレット（BGRで定義し、あとでLabに変換）
COLOR_PALETTE = {
    "ホワイト": [245, 245, 245],
    "ブラック": [20, 20, 20],
    "グレー": [128, 128, 128],
    "レッド": [40, 40, 200],      # OpenCVはBGR順
    "ネイビー": [120, 50, 20],
    "ブルー": [200, 100, 40],
    "グリーン": [40, 150, 40],
    "ブラウン": [30, 80, 120],
    "ベージュ": [180, 210, 230],
    "イエロー": [40, 220, 240],
    "ピンク": [180, 150, 220]
}

def analyze_clothing_color(img_rgb: Image.Image) -> dict:
    """
    背景を除外し、全ピクセルの多数決から「1位の色」と「2位の色」を返す
    """
    np_img = np.array(img_rgb)
    bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    mask = ~((bgr_img[:, :, 0] > 240) & (bgr_img[:, :, 1] > 240) & (bgr_img[:, :, 2] > 240))
    valid_pixels = bgr_img[mask]
    if len(valid_pixels) == 0:
        return {"first": "不明", "second": "-"}

    # パレットの色を Lab 空間に変換
    palette_names = list(COLOR_PALETTE.keys())
    palette_labs = []
    for color_name in palette_names:
        bgr_val = COLOR_PALETTE[color_name]
        palette_bgr = np.array([[bgr_val]], dtype=np.uint8)
        lab_val = cv2.cvtColor(palette_bgr, cv2.COLOR_BGR2Lab)[0][0].astype(float)
        palette_labs.append(lab_val)
    
    palette_array = np.array(palette_labs)

    # 全ピクセルを Lab に変換して距離を計算
    valid_labs = cv2.cvtColor(valid_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).reshape(-1, 3).astype(float)
    distances = np.linalg.norm(valid_labs[:, np.newaxis, :] - palette_array[np.newaxis, :, :], axis=2)
    min_indices = np.argmin(distances, axis=1)

    # 各パレットごとのピクセル数をカウント
    counts = np.bincount(min_indices, minlength=len(palette_names))

    # ピクセル数が多い順にインデックスをソートする（降順）
    sorted_indices = np.argsort(counts)[::-1]

    # 1位の色
    first_color = palette_names[sorted_indices[0]] if counts[sorted_indices[0]] > 0 else "不明"
    
    # 2位の色（もし2位の得票数が0より大きければそれを採用）
    second_color = "-"
    if len(sorted_indices) > 1 and counts[sorted_indices[1]] > 0:
        second_color = palette_names[sorted_indices[1]]

    return {
        "first": first_color,
        "second": second_color
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
        
        # 🌟 1位と2位の色を取得
        color_result = analyze_clothing_color(bg)

        final_path = os.path.join(TEMP_DIR, f"result_{model_name}.jpg")
        bg.save(final_path, "JPEG")

        return {
            "status": "success",
            "model": model_name,
            "detected_color": color_result["first"],     # 1位の色
            "second_color": color_result["second"],       # 2位の色
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