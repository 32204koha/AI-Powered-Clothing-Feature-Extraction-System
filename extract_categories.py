import csv
import json
import os
import time

CSV_PATH = r"C:\Users\Public\Documents\Kenkyu\AI-Powered-Clothing-Feature-Extraction-System\fashion-dataset\styles.csv"
OUTPUT_JSON = "category_prompts.json"

# データセットの英語カテゴリー名から洗練されたプロンプト群を自動生成
def generate_prompts_for_category(category_name: str) -> list:
    clean_name = category_name.lower().strip()
    return [
        f"a photo of a casual {clean_name}",
        f"a close-up photo of a {clean_name} apparel item",
        f"a fashion product photo of a {clean_name}"
    ]

def extract_all_categories():
    print("==================================================")
    print(" 👕 ファッションデータセット 全カテゴリー抽出ツール")
    print("==================================================")

    if not os.path.exists(CSV_PATH):
        print(f"⚠️ エラー: '{CSV_PATH}' が見つかりません。パスを確認してください。")
        return

    category_counts = {}
    sub_category_counts = {}
    total_rows = 0
    start_time = time.time()

    print(f"📂 '{CSV_PATH}' の解析を開始します...\n")

    # CSVから全行を読み込んで集計
    with open(CSV_PATH, mode='r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        
        fieldnames = reader.fieldnames or []
        print(f"📋 検出されたカラム一覧: {fieldnames}")

        for row in reader:
            total_rows += 1
            
            # 主要なカテゴリー情報（articleType）を取得
            article_type = row.get('articleType', '').strip()
            if article_type:
                category_counts[article_type] = category_counts.get(article_type, 0) + 1

            # サブカテゴリー（subCategory）も集計
            sub_cat = row.get('subCategory', '').strip()
            if sub_cat:
                sub_category_counts[sub_cat] = sub_category_counts.get(sub_cat, 0) + 1

            if total_rows % 10000 == 0:
                print(f"  ⏳ 読み込み中... 現在 {total_rows:,} 行目まで完了")

    elapsed_time = time.time() - start_time
    print("\n--------------------------------------------------")
    print(f" ✨ 解析完了！ (所要時間: {elapsed_time:.2f} 秒)")
    print(f" 📊 総解析レコード数: {total_rows:,} 行")
    print(f" 🏷️ 検出された articleType 数: {len(category_counts)} 種類")
    print(f" 📂 検出された subCategory 数: {len(sub_category_counts)} 種類")
    print("--------------------------------------------------\n")

    # 出現数が多い順にソート（※上限カットせず全件保持）
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    # 全カテゴリーをJSON構造化
    category_prompts_data = {}
    for cat_name, count in sorted_cats:
        category_prompts_data[cat_name] = {
            "count": count,
            "prompts": generate_prompts_for_category(cat_name)
        }

    # JSONに出力
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(category_prompts_data, f, indent=4, ensure_ascii=False)

    print(f"💾 全 {len(category_prompts_data)} 件のカテゴリー情報を '{OUTPUT_JSON}' に保存しました！")

if __name__ == "__main__":
    extract_all_categories()