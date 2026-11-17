from PIL import Image
import kagglehub
import os
from data_utils import load_and_clean_data
from model_utils import FashionAI

# APIキー（ご自身のものに書き換えてください）
API_KEY = "YOUR_API_KEY_HERE"

def main():
    print("1. データセットを準備しています（自動ダウンロードします。少し時間がかかります）...")
    # Kaggleから自動ダウンロード
    dataset_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-dataset")
    
    # 正しいパスを自動で設定
    csv_path = os.path.join(dataset_path, "fashion-dataset", "styles.csv")
    img_dir = os.path.join(dataset_path, "fashion-dataset", "images")
    
    print("2. データを読み込んでいます...")
    df = load_and_clean_data(csv_path, img_dir)
    
    print("3. AIを初期化しています...")
    ai = FashionAI(API_KEY)
    
    print("4. 解析を実行します...")
    target_id = 12497
    item = df[df['id'] == target_id].iloc[0]
    
    img = Image.open(item['image_path']).convert("RGB")
    desc = ai.get_description(img)
    result = ai.extract_json(desc, item.drop('image_path').to_dict())
    
    print(f"\n【解析結果】\n{result}")

if __name__ == "__main__":
    main()