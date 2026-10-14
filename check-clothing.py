from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import os
from PIL import Image
from rembg import remove

app = FastAPI()

# 既存のHTML（フロントエンド）からのリクエストを受け付けるためのCORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 1. アップロードされた画像ファイルの読み込み
    contents = await file.read()
    input_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # 2. rembg を使って背景を自動削除
    output_image_rgba = remove(input_image, model_name="u2netp")
    
    # 白背景のベースを作成し、背景を透過させた画像を重ね合わせる
    bg = Image.new("RGB", output_image_rgba.size, (255, 255, 255))
    bg.paste(output_image_rgba, mask=output_image_rgba.split()[3]) # Alphaチャンネルをマスクとして利用
    processed_image = bg

    # ==========================================
    # 【確認用】背景削除・白背景合成された画像を保存する
    # ==========================================
    os.makedirs("debug_outputs", exist_ok=True)
    debug_path = os.path.join("debug_outputs", f"processed_{file.filename}")
    processed_image.save(debug_path)
    print(f"背景削除後の画像を保存しました: {debug_path}")

    # --- ここから先に「色解析」や「CLIPによる種類判定」を記述していきます ---

    # 返却結果
    result = {
        "file": file.filename,
        "category": "t-shirt",
        "color": "white",
        "pattern": "solid",
        "confidence": 0.98
    }
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)