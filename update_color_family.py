import json
import os
import colorsys

PALETTE_FILE = r"C:\Users\Public\Documents\Kenkyu\AI-Powered-Clothing-Feature-Extraction-System\color_palette.json"

def get_color_family(color_name: str, bgr: list) -> str:
    """
    BGR値および色名から代表的な基本色(styles.csvのbaseColour互換)にマッピングする
    """
    clean_name = color_name.lower().strip()

    # 1. 名前に特定のキーが含まれる場合のダイレクトマッピング
    if "navy" in clean_name: return "Navy Blue"
    if "black" in clean_name or clean_name == "darkslategray": return "Black" if bgr[0] < 30 and bgr[1] < 30 and bgr[2] < 30 else "Grey"
    if "white" in clean_name: return "White"
    if "silver" in clean_name: return "Silver"
    if "gold" in clean_name: return "Gold"

    # 2. HSVによる動的判定
    b, g, r = bgr[0], bgr[1], bgr[2]
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)
    hue_deg = h * 360
    sat_pct = s * 100
    val_pct = v * 100

    # 無彩色・極暗色・極明度
    if val_pct < 15:
        return "Black"
    if sat_pct < 12:
        if val_pct > 85: return "White"
        if val_pct > 65: return "Silver"
        return "Grey"

    # 色相による分類
    if hue_deg < 15 or hue_deg >= 345:
        return "Red" if val_pct > 30 else "Maroon"
    elif hue_deg < 45:
        if sat_pct < 40 and val_pct > 70: return "Beige"
        return "Orange" if val_pct > 60 else "Brown"
    elif hue_deg < 70:
        return "Yellow"
    elif hue_deg < 160:
        return "Green" if val_pct > 35 else "Olive"
    elif hue_deg < 200:
        return "Blue" if val_pct > 60 else "Navy Blue"
    elif hue_deg < 260:
        return "Navy Blue" if val_pct < 45 else "Blue"
    elif hue_deg < 310:
        return "Purple"
    elif hue_deg < 345:
        return "Pink"

    return "Grey"

def update_palette_with_color_family():
    if not os.path.exists(PALETTE_FILE):
        print(f"❌ エラー: {PALETTE_FILE} が見つかりません。")
        return

    with open(PALETTE_FILE, "r", encoding="utf-8") as f:
        palette = json.load(f)

    updated_count = 0
    for name, data in palette.items():
        bgr = data.get("bgr", [128, 128, 128])
        family = get_color_family(name, bgr)
        data["color_family"] = family
        updated_count += 1

    with open(PALETTE_FILE, "w", encoding="utf-8") as f:
        json.dump(palette, f, indent=4, ensure_ascii=False)

    print(f"✨ {updated_count} 種類の色に `color_family` を付与・更新しました。")

if __name__ == "__main__":
    update_palette_with_color_family()