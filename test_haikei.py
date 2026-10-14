from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import io
import os
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

# 🌟 主要な4つのモデルを起動時にすべてメモリに常駐させる
sessions = {}
TARGET_MODELS = ["u2net", "u2netp", "isnet-general-use", "bria-rmbg"]

print("🔄 比較用の各種AIモデルをメモリにロード中...")
for name in TARGET_MODELS:
    try:
        sessions[name] = new_session(name)
        print(f"  - ロード成功: {name}")
    except Exception as e:
        print(f"  - ロード失敗 ({name}): {e}")

@app.post("/test-remove-bg")
async def test_remove_bg(
    file: UploadFile = File(...),
    model_name: str = Form("u2net")
):
    try:
        contents = await file.read()
        input_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 選択されたモデルのセッションを取得（なければ標準の u2net）
        session = sessions.get(model_name, sessions.get("u2net"))
        
        # 背景削除を実行
        output_image_rgba = remove(input_image, session=session)
        
        # 白背景に合成
        bg = Image.new("RGB", output_image_rgba.size, (255, 255, 255))
        bg.paste(output_image_rgba, mask=output_image_rgba.split()[3])
        
        # モデルごとのファイル名で保存
        final_path = os.path.join(TEMP_DIR, f"result_{model_name}.jpg")
        bg.save(final_path, "JPEG")

        return {
            "status": "success",
            "model": model_name,
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