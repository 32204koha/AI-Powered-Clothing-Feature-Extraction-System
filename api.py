from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import json
from model_utils import FashionAI
from scoring import calculate_item_score, calculate_total_score
import os
import uuid
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# ご自身のAPIキーを設定してください
load_dotenv()
API_KEY = os.getenv("GCP_API_KEY")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開発中なので一旦すべて許可
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("サーバー起動中... AIモデルを読み込んでいます（少し時間がかかります）...")
# サーバー起動時に1回だけAIをスタンバイさせておく
ai = FashionAI(API_KEY)
print("AIのスタンバイが完了しました！リクエストを受け付けられます。")

# ==========================================
# 1. データ保存用の準備
# ==========================================

IMAGE_DIR = "saved_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# 全体診断用：サーバーが起動している間だけ記憶するリスト
closet_db = []

# ==========================================
# 2. 画像アップロードと個別解析処理 (/analyze)
# ==========================================


@app.post("/analyze")
async def analyze_clothing(
    file: UploadFile = File(...),      
    survey_result: str = Form(...)     
):
    print(f"\n--- 新しいリクエスト ---")
    print(f"1. 画像({file.filename})とアンケート({survey_result})を受信しました！")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # ランダムなファイル名(UUID)を生成して拡張子.jpgで保存
    unique_filename = f"{uuid.uuid4()}.jpg"
    image_save_path = os.path.join(IMAGE_DIR, unique_filename)
    image.save(image_save_path, "JPEG")
    # フロントエンドからアクセスする時のURL
    image_url = f"/images/{unique_filename}"
    print(f"-> 画像を保存しました: {image_save_path}")
    # ---------------------------------------------
    
    print("2. Geminiで画像とデータを解析しています...")
    meta_data = {"survey_result": survey_result}
    
    extracted_text = ai.analyze_with_gemini(image, meta_data)
    
    # 複数アイテムが入るように空のリストを準備
    extracted_items = []
    try:
        # Markdownのゴミを取ってから変換
        clean_text = extracted_text.strip().removeprefix("```json").removesuffix("```").strip()
        parsed_data = json.loads(clean_text)
        
        # AIが1着だけ（ {} ）で返してきた場合は、強制的にリスト（ [{}] ）に包む保険
        if isinstance(parsed_data, dict):
            extracted_items = [parsed_data]
        elif isinstance(parsed_data, list):
            extracted_items = parsed_data
            
    except Exception as e:
        print(f"JSONの変換に失敗しましたが続行します: {e}")
        extracted_items = [{"raw_text": extracted_text}]
    
    print("3. 各アイテムのスコアを計算し、登録します...")
    processed_items = []
    
    # AIが見つけた服の数だけループして、1着ずつスコア計算と登録を行う
    for item in extracted_items:
        # 1着ごとのスコアを計算
        final_score = calculate_item_score(item, survey_result)
        
        # データの中に共通の情報を追加
        item["image_url"] = image_url
        item["survey_result"] = survey_result
        item["score"] = final_score
        
        # 1着ずつクローゼットDBに追加（トップス、パンツなどが別々のデータとして入る）
        closet_db.append(item)
        processed_items.append(item)

    # DBに追加された最新のリストを history.json に上書き保存する
    try:
        with open("history.json", "w", encoding="utf-8") as f:
            json.dump(closet_db, f, ensure_ascii=False, indent=4)
        print("履歴を history.json に保存しました。")
    except Exception as e:
        print(f"履歴の保存に失敗しました: {e}")
    
    # 最終的なレスポンスデータを作成
    result_data = {
        "status": "success",
        "message": f"{len(processed_items)}着のアイテムを認識・登録しました",
        "extracted_items": processed_items,
        "image_url": image_url
    }

    return result_data
    
    print("4. 結果を history.json と メモリ(closet_db) に保存しています...")
    # ファイルへのバックアップ保存
    with open("history.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
        
    # 全体診断(diagnose)のために、メモリにも追加
    # アンケート結果も属性データに混ぜておく
    extracted_attributes["survey_result"] = survey_result
    
    print("5. 結果をフロントエンドに返却します！")
    return result_data

# ==========================================
# 3. 全体診断と断捨離アドバイス処理 (/diagnose)
# ==========================================


@app.get("/diagnose")
async def diagnose_closet():
    print(f"\n--- 診断リクエスト受信 ---")
    if len(closet_db) == 0:
        return {"diagnosis": "まだ服が登録されていません。まずは画像をアップロードしてください。", "score": 0}
    
    print("1. 仮のルールベースで全体スコアを計算しています...")
    mock_score = mock_score = calculate_total_score(closet_db)
    
    print("2. Geminiに全体アドバイスを依頼しています...")
    diagnosis_text = ai.diagnosis_with_gemini(closet_db)
    
    print("3. 診断結果を返却します！")
    return {
        "score": mock_score,
        "diagnosis": diagnosis_text
    }

# --- 4. 登録された服の一覧を取得するエンドポイント ---
@app.get("/closet")
async def get_closet():
    # 登録されている全データのリストを返す
    return {"closet_items": closet_db, "total_count": len(closet_db)}

@app.get("/gallery")
async def render_gallery():
    return FileResponse("gallery.html")