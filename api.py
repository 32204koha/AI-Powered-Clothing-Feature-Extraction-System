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
from datetime import datetime
from fastapi.responses import FileResponse

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

@app.get("/")
async def read_index():
    return FileResponse("index.html")

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
    survey_result: str = Form(...),  
    entry_type: str = Form(...),           # "worn_today"(今日着る) or "inventory"(持っている服)
    days_since_last_worn: str = Form(default="不明") # 何日ぶりか（持っている服の登録時は空っぽ）   
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

    # 画像をPIL形式で開く
    original_img = Image.open(io.BytesIO(image_bytes))
    img_width, img_height = original_img.size
    
    print("3. 各アイテムのスコアを計算し、登録します...")
    processed_items = []

    current_date = datetime.now().strftime("%Y-%m-%d")

    
    # AIが見つけた服の数だけループして、1着ずつスコア計算と登録を行う
    for item in extracted_items:
        # 1着ごとのスコアを計算
        final_score = calculate_item_score(item, survey_result)

        # --- 切り抜き（ズーム）処理 ---
        item_image_url = image_url  # デフォルトは全身写真のURL
        
        box = item.get("box_2d")
        if box and len(box) == 4:
            try:
                # 0~1000の座標をピクセル単位に変換
                ymin, xmin, ymax, xmax = box
                left = int(xmin / 1000 * img_width)
                top = int(ymin / 1000 * img_height)
                right = int(xmax / 1000 * img_width)
                bottom = int(ymax / 1000 * img_height)

                # 服がぴったりギリギリにならないよう、少し余白（5%）を持たせる
                pad_w = int((right - left) * 0.05)
                pad_h = int((bottom - top) * 0.05)
                left = max(0, left - pad_w)
                top = max(0, top - pad_h)
                right = min(img_width, right + pad_w)
                bottom = min(img_height, bottom + pad_h)

                # 画像を切り抜く
                cropped_img = original_img.crop((left, top, right, bottom))

                # 切り抜いた画像を新しいファイルとして保存
                cropped_filename = f"crop_{uuid.uuid4()}.jpg"
                cropped_path = os.path.join(IMAGE_DIR, cropped_filename)
                
                # RGB変換してJPEGで保存
                if cropped_img.mode != 'RGB':
                    cropped_img = cropped_img.convert('RGB')
                cropped_img.save(cropped_path, "JPEG")

                # 個別アイテム用画像URLに差し替え
                item_image_url = f"/images/{cropped_filename}"
                print(f"  -> {item.get('category', 'アイテム')} の切り抜き画像を保存しました: {cropped_filename}")
            except Exception as e:
                print(f"切り抜き処理に失敗しました (全体画像を使用します): {e}")
        
        # データの中に共通の情報を追加
        item["image_url"] = item_image_url
        item["original_image_url"] = image_url
        item["survey_result"] = survey_result
        item["score"] = final_score

        item["registered_date"] = current_date
        item["entry_type"] = entry_type
        item["days_since_last_worn"] = days_since_last_worn or "不明"
        
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

# -- 検索機能
@app.get("/search")
def search_closet(q: str = None):
    """キーワードでクローゼット内のアイテムを検索する"""
    # 検索キーワードが空の場合は、すべてのアイテムを返す
    if not q:
        return {"status": "success", "results": closet_db, "count": len(closet_db)}
    
    search_results = []
    # キーワードを小文字にしておく（大文字・小文字の区別をなくすため）
    keyword = q.lower()
    
    for item in closet_db:
        # アイテムが持っている全ての情報（色、カテゴリ、状態など）を一つの文字列にまとめる
        item_text = " ".join([str(val) for val in item.values()]).lower()
        
        # 検索キーワードが含まれていれば結果リストに追加
        if keyword in item_text:
            search_results.append(item)
            
    return {
        "status": "success",
        "results": search_results,
        "count": len(search_results)
    }