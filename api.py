from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import json
from model_utils import FashionAI
import os
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

# 全体診断用：サーバーが起動している間だけ記憶するリスト
closet_db = []

# ==========================================
# 2. 画像アップロードと個別解析処理 (/analyze)
# ==========================================

# （既存の）1着ごとのスコア計算関数
def calculate_score(attributes: dict, survey_data: str) -> int:
    score = 50 
    
    if isinstance(attributes, dict):
        if attributes.get("gender") == "Men":
            score += 10
            
    if survey_data == "好き":
        score += 30
    elif survey_data == "普通":
        score += 10
        
    return score

@app.post("/analyze")
async def analyze_clothing(
    file: UploadFile = File(...),      
    survey_result: str = Form(...)     
):
    print(f"\n--- 新しいリクエスト ---")
    print(f"1. 画像({file.filename})とアンケート({survey_result})を受信しました！")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    print("2. Geminiで画像とデータを解析しています...")
    meta_data = {"survey_result": survey_result}
    
    extracted_text = ai.analyze_with_gemini(image, meta_data)
    
    extracted_attributes = {}
    try:
        # Markdownのゴミを取ってから変換
        clean_text = extracted_text.strip().removeprefix("```json").removesuffix("```").strip()
        extracted_attributes = json.loads(clean_text)
    except Exception as e:
        print(f"JSONの変換に失敗しましたが続行します: {e}")
        extracted_attributes = {"raw_text": extracted_text}
    
    print("3. 個別スコアを計算しています...")
    final_score = calculate_score(extracted_attributes, survey_result)
    
    result_data = {
        "status": "success",
        "extracted_attributes": extracted_attributes,
        "calculated_score": final_score,
        "survey_received": survey_result
    }
    
    print("4. 結果を history.json と メモリ(closet_db) に保存しています...")
    # ファイルへのバックアップ保存
    with open("history.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
        
    # 全体診断(diagnose)のために、メモリにも追加
    # アンケート結果も属性データに混ぜておく
    extracted_attributes["survey_result"] = survey_result
    closet_db.append(extracted_attributes)
    
    print("5. 結果をフロントエンドに返却します！")
    return result_data

# ==========================================
# 3. 全体診断と断捨離アドバイス処理 (/diagnose)
# ==========================================

# 後のルールベースAIに置き換えるための「仮の全体スコア計算関数」
def calculate_mock_total_score(closet_data):
    total_items = len(closet_data)
    if total_items == 0:
        return 100
        
    score = 100
    for item in closet_data:
        # アンケートによる減点
        if item.get("survey_result") == "嫌い":
            score -= 20
        elif item.get("survey_result") == "保留":
            score -= 10
            
        # 状態による減点
        if item.get("condition") == "使用感あり":
            score -= 5
            
    return max(0, min(100, score))

@app.get("/diagnose")
async def diagnose_closet():
    print(f"\n--- 診断リクエスト受信 ---")
    if len(closet_db) == 0:
        return {"diagnosis": "まだ服が登録されていません。まずは画像をアップロードしてください。", "score": 0}
    
    print("1. 仮のルールベースで全体スコアを計算しています...")
    mock_score = calculate_mock_total_score(closet_db)
    
    print("2. Geminiに全体アドバイスを依頼しています...")
    diagnosis_text = ai.diagnosis_with_gemini(closet_db)
    
    print("3. 診断結果を返却します！")
    return {
        "score": mock_score,
        "diagnosis": diagnosis_text
    }