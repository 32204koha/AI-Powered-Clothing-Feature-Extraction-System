import pandas as pd
import os

def load_and_clean_data(csv_path, images_dir):
    df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
    df = df.dropna(subset=['productDisplayName'])
    df['image_path'] = df['id'].apply(lambda x: os.path.join(images_dir, str(x) + '.jpg'))
    
    # フィルタリング
    allowed = ['Apparel', 'Footwear', 'Accessories']
    df = df[df['masterCategory'].isin(allowed)].copy()
    
    # 画像の存在チェック
    existing_images = set(os.listdir(images_dir))
    df = df[df['id'].astype(str).apply(lambda x: f"{x}.jpg" in existing_images)]
    return df