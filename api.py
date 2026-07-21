from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
import io
import json
from model_utils import FashionAI
import os
from dotenv import load_dotenv

# ご自身のAPIキーを設定してください
load_dotenv()
API_KEY = os.getenv("GCP_API_KEY")

app = FastAPI()

print("サーバー起動中... AIモデルを読み込んでいます（少し時間がかかります）...")
# サーバー起動時に1回だけAIをスタンバイさせておく
ai = FashionAI(API_KEY)
print("AIのスタンバイが完了しました！リクエストを受け付けられます。")

def calculate_score(attributes: dict, survey_data: str) -> int:
    score = 50 
    
    # AIが抽出したJSONの中身（辞書型）を使ってスコアを計算
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
    print(f"1. 画像({file.filename})とアンケート({survey_result})を受信しました！")
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    print("2. Geminiで画像とデータを解析しています...")
    meta_data = {"survey_result": survey_result}
    
    # 直接Geminiに画像を読ませる
    extracted_text = ai.analyze_with_gemini(image, meta_data)
    
    # AIの返答(文字列)をPythonの辞書型(dict)に変換
    extracted_attributes = {}
    try:
        # Markdownの ```json ... ``` などのゴミを取ってから変換
        clean_text = extracted_text.strip().removeprefix("```json").removesuffix("```").strip()
        extracted_attributes = json.loads(clean_text)
    except Exception as e:
        print(f"JSONの変換に失敗しましたが続行します: {e}")
        extracted_attributes = {"raw_text": extracted_text}
    
    print("3. スコアを計算しています...")
    final_score = calculate_score(extracted_attributes, survey_result)
    
    # 最終的な結果データ
    result_data = {
        "status": "success",
        "extracted_attributes": extracted_attributes,
        "calculated_score": final_score,
        "survey_received": survey_result
    }
    
    print("4. 結果を history.json に保存しています...")
    # ★ 履歴をファイルに保存する機能（レベル1）を追加！
    with open("history.json", "a", encoding="utf-8") as f:
        f.write(json.dumps(result_data, ensure_ascii=False) + "\n")
    
    print("5. 結果をスマホアプリに返却します！")
    return result_data