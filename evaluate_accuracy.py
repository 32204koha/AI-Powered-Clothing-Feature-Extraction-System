import os
import json
import time
import re
import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.getenv("GCP_API_KEY") or os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("APIキーが設定されていません。.envファイルを確認してください。")

# ==========================================
# 設定情報
# ==========================================
# 実在するモデル名に変更（標準的なGemini 2.5 Flashまたは2.0 Flashを推奨）
MODEL_NAME = "gemini-3.8-flash"

DATASET_DIR = r"C:\Users\Public\Documents\Kenkyu\AI-Powered-Clothing-Feature-Extraction-System\fashion-dataset"
CSV_PATH = os.path.join(DATASET_DIR, "styles.csv")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

NUM_SAMPLES = 50
REQUEST_INTERVAL = 4.5  # 15 RPM制限回避用

client = genai.Client(api_key=API_KEY)


# ==========================================
# レスポンス構造の定義 (Pydantic Schema)
# API側に返却フォーマットを厳密に保証させるため
# ==========================================
class FashionItem(BaseModel):
    category: str = Field(description="Main classification (Apparel / Footwear / Accessories / OR 'not clothes')")
    sub_category: str = Field(description="Sub-category (e.g., Topwear, Bottomwear, Dress, Shoes, Bags, Headwear, Flip Flops)")
    article_type: str = Field(description="Specific item name (e.g., Tshirts, Shirts, Jackets, Kurtas, Dresses, Jeans, Trousers, Sneakers, Handbags)")
    color: str = Field(description="Primary base color identified from the image")
    secondary_color: str = Field(description="Secondary or accent color if applicable, otherwise 'None'")
    gender_target: str = Field(description="Target demographic (Men / Women / Unisex / Boys / Girls)")
    style_taste: str = Field(description="Aesthetic style (e.g., Casual, Streetwear, Formal, Minimalist, Ethnic, Sporty, Preppy)")
    season: str = Field(description="Best fitting season (Summer / Winter / Spring / Autumn / All Season)")
    pattern: str = Field(description="Fabric pattern (e.g., Solid, Striped, Graphic Print, Checked, Floral, Polka Dot, Distressed)")
    scene_usage: str = Field(description="Usage scene (e.g., Casual, Formal, Work, Sports, Party, Loungewear, School)")
    material_texture: str = Field(description="Perceived material/texture (e.g., Denim, Cotton, Leather, Knit, Canvas, Synthetic, Silk, Mesh)")
    brand: str = Field(description="Brand name read strictly from visible logos or tags. Set as 'Unknown' if not clearly visible")


# ==========================================
# 英語プロンプトテンプレート
# ==========================================
LLM_PROMPT_TEMPLATE = """
You are an expert AI fashion stylist and automated image analyst specializing in deep clothing attribute extraction.
Analyze the attached image and the provided dataset metadata to accurately extract detailed fine-grained fashion attributes.

【INPUT METADATA】
{meta}

【INSTRUCTIONS】
1. Carefully inspect the image using the provided metadata as additional contextual ground truth.
2. If the item in the image is NOT wearable clothing, footwear, or fashion accessories (e.g., perfume, deodorant, cosmetics, home decor, electronics), set "category" strictly to "not clothes".
3. Identify each distinct fashion item present in the image.
4. Extract detailed fashion attributes including main/secondary color, material/texture, style taste, pattern, scene usage, target demographic, and visible brand.

【STRICT RULES】
1. Do not omit any fields in the output schema.
2. Be highly accurate with colors and material textures, carefully accounting for lighting conditions.
3. NEVER guess brand names based on style or design alone. If a logo or tag text is not clearly readable, strictly output "Unknown".
"""


# ==========================================
# 解析関数 (リトライロジック・型安全対応)
# ==========================================
def analyze_clothing_detailed(image_path: str, meta_str: str, max_retries: int = 5) -> dict:
    if not os.path.exists(image_path):
        print(f" ⚠️ 画像ファイルが存在しません: {image_path}")
        return {}

    prompt = LLM_PROMPT_TEMPLATE.format(meta=meta_str)

    # 画像の安全なオープンとRGB変換（RGBAやPaletteモードによるエラー防止）
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            
            for attempt in range(max_retries):
                try:
                    # Structured Outputs (Pydantic Schema) を適用
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[img_rgb, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=list[FashionItem]  # 配列形式を保証
                        )
                    )

                    if not response.text:
                        print(" ⚠️ APIからの応答テキストが空です。")
                        continue

                    # JSONとして安全にロード
                    parsed_data = json.loads(response.text)

                    if isinstance(parsed_data, list) and len(parsed_data) > 0:
                        return parsed_data[0]
                    elif isinstance(parsed_data, dict):
                        return parsed_data
                    
                    return {}

                except Exception as e:
                    error_str = str(e)
                    # 429 (Rate Limit / Quota Exceeded) や 503 (Service Unavailable) のリトライ
                    if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str or "UNAVAILABLE" in error_str:
                        wait_time = (2 ** attempt) * 5
                        print(f" ⚠️ レート制限または一時エラー ({e})。 {wait_time}秒待機後に再試行... ({attempt+1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        print(f" ⚠️ API呼び出しエラー: {e}")
                        if attempt == max_retries - 1:
                            return {}
                        time.sleep(3)

    except Exception as img_err:
        print(f" ⚠️ 画像処理エラー: {img_err}")
        return {}

    return {}


# ==========================================
# メイン処理
# ==========================================
def main():
    print(f"Gemini API ('{MODEL_NAME}') を使用して詳細な属性抽出・精度検証を開始します...")
    
    if not os.path.exists(CSV_PATH):
        print(f" ❌ CSVファイルが見つかりません: {CSV_PATH}")
        return

    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    
    # ランダムサンプリング
    sample_df = df.sample(n=min(NUM_SAMPLES, len(df)), random_state=42).reset_index(drop=True)

    results = []
    category_correct = 0
    color_correct = 0
    total_samples = 0
    not_clothes_count = 0
    clothes_valid_count = 0
    elapsed_times = []

    print(f"--- 評価開始 (全 {len(sample_df)} 件) ---")

    for idx, row in sample_df.iterrows():
        # image_idがfloat型になる現象を防ぐため、int変換してから文字列化
        try:
            image_id = str(int(row['id']))
        except (ValueError, TypeError):
            image_id = str(row['id'])

        image_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")

        # メタデータの構築
        gt_article_type = str(row.get('articleType', '')).strip().lower()
        gt_color = str(row.get('baseColour', '')).strip().lower()
        
        meta_str = (
            f"ID: {image_id}, "
            f"Gender: {row.get('gender', '')}, "
            f"MasterCategory: {row.get('masterCategory', '')}, "
            f"SubCategory: {row.get('subCategory', '')}, "
            f"ArticleType: {gt_article_type}, "
            f"BaseColour: {gt_color}, "
            f"Season: {row.get('season', '')}, "
            f"Usage: {row.get('usage', '')}, "
            f"ProductDisplayName: {row.get('productDisplayName', '')}"
        )

        start_time = time.time()
        ai_output = analyze_clothing_detailed(image_path, meta_str)
        elapsed_time = time.time() - start_time
        elapsed_times.append(elapsed_time)

        ai_category = str(ai_output.get('category', '')).strip().lower()
        ai_article_type = str(ai_output.get('article_type', '')).strip().lower()
        ai_color = str(ai_output.get('color', '')).strip().lower()

        is_not_clothes = (ai_category == "not clothes")
        is_cat_match = False
        is_col_match = False

        if is_not_clothes:
            not_clothes_count += 1
        else:
            if ai_article_type and gt_article_type:
                is_cat_match = (gt_article_type in ai_article_type) or (ai_article_type in gt_article_type)
            if ai_color and gt_color:
                is_col_match = (gt_color in ai_color) or (ai_color in gt_color)

            if is_cat_match: category_correct += 1
            if is_col_match: color_correct += 1
            clothes_valid_count += 1

        total_samples += 1

        # 解析結果の格納
        res = {
            "id": image_id,
            "gt_article_type": gt_article_type,
            "ai_article_type": ai_article_type,
            "cat_match": is_cat_match,
            "gt_color": gt_color,
            "ai_color": ai_color,
            "col_match": is_col_match,
            "ai_category": ai_category,
            "sub_category": ai_output.get('sub_category', ''),
            "secondary_color": ai_output.get('secondary_color', ''),
            "gender_target": ai_output.get('gender_target', ''),
            "style_taste": ai_output.get('style_taste', ''),
            "season": ai_output.get('season', ''),
            "pattern": ai_output.get('pattern', ''),
            "scene_usage": ai_output.get('scene_usage', ''),
            "material_texture": ai_output.get('material_texture', ''),
            "brand": ai_output.get('brand', 'Unknown'),
            "processing_time_sec": round(elapsed_time, 2)
        }
        results.append(res)

        if is_not_clothes:
            print(f"[{total_samples}/{len(sample_df)}] ID:{image_id} | 🚫【Not Clothes】(GT: '{gt_article_type}') | Time: {elapsed_time:.2f}s")
        else:
            print(f"[{total_samples}/{len(sample_df)}] ID:{image_id} | "
                  f"Type: GT '{gt_article_type}' vs AI '{ai_article_type}' ({'OK' if is_cat_match else 'NG'}) | "
                  f"Color: GT '{gt_color}' vs AI '{ai_color}' ({'OK' if is_col_match else 'NG'}) | "
                  f"Time: {elapsed_time:.2f}s")

        time.sleep(REQUEST_INTERVAL)

    # 最終集計結果
    if total_samples > 0:
        cat_acc = (category_correct / clothes_valid_count) * 100 if clothes_valid_count > 0 else 0
        col_acc = (color_correct / clothes_valid_count) * 100 if clothes_valid_count > 0 else 0
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0

        print("\n==========================================")
        print("     詳細属性評価レポート (Gemini)       ")
        print("==========================================")
        print(f"使用モデル         : {MODEL_NAME}")
        print(f"評価試行数(全体)   : {total_samples} 件")
        print(f"🚫服以外として除外 : {not_clothes_count} 件")
        print("------------------------------------------")
        if clothes_valid_count > 0:
            print(f"👕服として評価     : {clothes_valid_count} 件")
            print(f"品目名(Type)正解率 : {cat_acc:.2f}% ({category_correct}/{clothes_valid_count})")
            print(f"色抽出(Color)正解率: {col_acc:.2f}% ({color_correct}/{clothes_valid_count})")
        print("------------------------------------------")
        print(f"平均処理時間       : {avg_time:.2f} 秒 / 件")
        print("==========================================")

        out_csv = "evaluation_results_gemini_detailed.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"詳細属性の解析結果を '{out_csv}' に保存しました。")


if __name__ == "__main__":
    main()