import os
import json
import time
import re
import pandas as pd
import ollama

# ==========================================
# 設定情報
# ==========================================
# 使用するOllamaモデル名 ('llava' または 'llama3.2-vision' など)
OLLAMA_MODEL = "llama3.2-vision"

# fashion-dataset のパス
DATASET_DIR = r"C:\Users\Public\Documents\Kenkyu\AI-Powered-Clothing-Feature-Extraction-System\fashion-dataset"
CSV_PATH = os.path.join(DATASET_DIR, "styles.csv")
IMAGES_DIR = os.path.join(DATASET_DIR, "images")

NUM_SAMPLES = 20
# ローカル動作のためAPI制限用の待ち時間はゼロ（0.5秒程度）でOK
REQUEST_INTERVAL = 0.5


# ==========================================
# Ollama用 解析関数（修正版）
# ==========================================
def analyze_clothing_for_eval(image_path: str) -> dict:
    prompt = """
    You are a professional fashion stylist AI.
    Based on the attached image, extract the attributes and the location of the clothing within the image (bounding box [ymin, xmin, ymax, xmax] normalized coordinates from 0-1000), and output strictly in the JSON format below.

    【CRITICAL】
    1. Return a list (array) of items. Even if only one piece of clothing is present, return a list containing a single element.
    2. If the image is NOT clothing or wearable fashion accessories (e.g., perfume, deodorant, body mist, cosmetics, interior goods, etc.), set the "category" strictly to "not clothes".

    【Output Format】
    [
        {
        "category": "Choose the most accurate one from: Tshirts, Shirts, Jackets, Sweaters, Sweatshirts, Kurtas, Dresses, Trousers, Jeans, Capris, Shorts, Skirts, Underwear, Briefs, Trunk, Nightdress, Accessories, Shoes, Bags, Handbags, Wallets, Belts, Caps, Headband, OR 'not clothes'",
        "color": "Accurate color determined from the image",
        "style_taste": "Simple/Street/Casual/etc.",
        "season": "Summer/Winter/All season/etc.",
        "pattern": "Solid/Striped/Checked/Printed/etc.",
        "scene": "Work/Casual/Loungewear/Sports/School/etc.",
        "brand": "Brand name read from visible logos or tags. If unidentifiable, set as 'Unknown'",
        "box_2d": [100, 200, 500, 800]
        }
    ]

    【Strict Rules】
    1. Do not omit any fields.
    2. Output raw JSON only.
    """
    
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }],
            format='json'
        )
        
        text = response['message']['content'].strip()
        
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match: 
            text = match.group(0)
        else:
            match_dict = re.search(r'\{.*\}', text, re.DOTALL)
            if match_dict: 
                text = match_dict.group(0)

        parsed_data = json.loads(text)
        
        # 配列で返ってきた場合、最初の要素が辞書(dict)か確認
        if isinstance(parsed_data, list) and len(parsed_data) > 0:
            if isinstance(parsed_data[0], dict):
                return parsed_data[0]
            return {}
        elif isinstance(parsed_data, dict):
            return parsed_data
        else: 
            return {}
            
    except Exception:
        # パース失敗や通信エラー時は空の辞書を返す
        return {}


# ==========================================
# メイン評価処理
# ==========================================
def main():
    print(f"モデル '{OLLAMA_MODEL}' を使用して評価を行います...")
    print(f"styles.csv を読み込んでいます: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH, on_bad_lines='skip')
    
    sample_df = df.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)
    
    results = []
    category_correct = 0
    color_correct = 0
    total_samples = 0
    not_clothes_count = 0
    clothes_valid_count = 0
    elapsed_times = []

    print(f"--- 評価開始 (全 {NUM_SAMPLES} 件) ---")

    for idx, row in sample_df.iterrows():
        image_id = row['id']
        image_name = f"{image_id}.jpg"
        image_path = os.path.join(IMAGES_DIR, image_name)
        
        if not os.path.exists(image_path): continue

        gt_category = str(row.get('articleType', '')).strip().lower()
        gt_color = str(row.get('baseColour', '')).strip().lower()

        try:
            start_time = time.time()
            
            # ローカルAIの実行
            ai_output = analyze_clothing_for_eval(image_path)

            # 【追加】ai_outputが辞書型でない場合は空の辞書にする
            if not isinstance(ai_output, dict):
                ai_output = {}
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_times.append(elapsed_time)
            
            ai_category = str(ai_output.get('category', '')).strip().lower()
            ai_color = str(ai_output.get('color', '')).strip().lower()

            is_not_clothes = (ai_category == "not clothes")
            is_cat_match = False
            is_col_match = False

            if is_not_clothes:
                not_clothes_count += 1
            else:
                if ai_category and gt_category:
                    is_cat_match = (gt_category in ai_category) or (ai_category in gt_category)
                
                if ai_color and gt_color:
                    is_col_match = (gt_color in ai_color) or (ai_color in gt_color)

                if is_cat_match: category_correct += 1
                if is_col_match: color_correct += 1
                clothes_valid_count += 1
                
            total_samples += 1

            res = {
                "id": image_id,
                "gt_category": gt_category,
                "ai_category": ai_category,
                "is_not_clothes": is_not_clothes,
                "cat_match": is_cat_match,
                "gt_color": gt_color,
                "ai_color": ai_color,
                "col_match": is_col_match,
                "processing_time_sec": round(elapsed_time, 2)
            }
            results.append(res)
            
            if is_not_clothes:
                print(f"[{total_samples}/{NUM_SAMPLES}] ID:{image_id} | 🚫【服以外として除外】(GT: '{gt_category}') | Time: {elapsed_time:.2f}s")
            else:
                print(f"[{total_samples}/{NUM_SAMPLES}] ID:{image_id} | "
                      f"Cat: GT '{gt_category}' vs AI '{ai_category}' ({'OK' if is_cat_match else 'NG'}) | "
                      f"Color: GT '{gt_color}' vs AI '{ai_color}' ({'OK' if is_col_match else 'NG'}) | "
                      f"Time: {elapsed_time:.2f}s")

        except Exception as e:
            print(f"[{idx+1}/{NUM_SAMPLES}] エラー発生 (ID:{image_id}): {e}")

        time.sleep(REQUEST_INTERVAL)

    # ==========================================
    # 集計とレポート表示
    # ==========================================
    if total_samples > 0:
        cat_acc = (category_correct / clothes_valid_count) * 100 if clothes_valid_count > 0 else 0
        col_acc = (color_correct / clothes_valid_count) * 100 if clothes_valid_count > 0 else 0
        
        min_time = min(elapsed_times) if elapsed_times else 0
        max_time = max(elapsed_times) if elapsed_times else 0
        avg_time = sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0

        print("\n==========================================")
        print("          精度評価結果レポート (Ollama)   ")
        print("==========================================")
        print(f"使用モデル       : {OLLAMA_MODEL}")
        print(f"評価試行数(全体) : {total_samples} 件")
        print(f"🚫服以外として除外: {not_clothes_count} 件")
        print("------------------------------------------")
        if clothes_valid_count > 0:
            print(f"👕服として評価   : {clothes_valid_count} 件")
            print(f"カテゴリ正解率   : {cat_acc:.2f}% ({category_correct}/{clothes_valid_count})")
            print(f"色抽出正解率     : {col_acc:.2f}% ({color_correct}/{clothes_valid_count})")
        print("------------------------------------------")
        print("【処理時間統計】")
        print(f"最短処理時間     : {min_time:.2f} 秒")
        print(f"最長処理時間     : {max_time:.2f} 秒")
        print(f"平均処理時間     : {avg_time:.2f} 秒")
        print("==========================================")

        res_df = pd.DataFrame(results)
        res_df.to_csv("evaluation_results_ollama.csv", index=False, encoding="utf-8-sig")
        print("詳細結果を 'evaluation_results_ollama.csv' に保存しました。")

if __name__ == "__main__":
    main()