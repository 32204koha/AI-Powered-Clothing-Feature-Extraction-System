import json
import os

try:
    import webcolors
except ImportError:
    print("⚠️ 'webcolors' ライブラリが必要です。以下のコマンドでインストールしてください:")
    print("   pip install webcolors")
    exit()

OUTPUT_JSON = r"C:\Users\Public\Documents\Kenkyu\AI-Powered-Clothing-Feature-Extraction-System\color_palette.json"

def get_css3_color_map():
    """webcolorsのバージョン差分を吸収してカラー辞書を取得"""
    # 1. 新しいバージョン (v24.0+) の取得方法
    if hasattr(webcolors, "names"):
        try:
            color_names = webcolors.names("css3")
            return {name: webcolors.name_to_hex(name, spec="css3") for name in color_names}
        except Exception:
            pass

    # 2. 中期バージョンの取得方法
    if hasattr(webcolors, "CSS3_NAMES_TO_HEX"):
        return webcolors.CSS3_NAMES_TO_HEX

    if hasattr(webcolors, "CSS3_HEX_TO_NAMES"):
        return {v: k for k, v in webcolors.CSS3_HEX_TO_NAMES.items()}

    # 3. フォールバック: html5 / css21 などの定数
    for attr in ["HTML5_NAMES_TO_HEX", "CSS21_NAMES_TO_HEX"]:
        if hasattr(webcolors, attr):
            return getattr(webcolors, attr)

    return {}

def expand_color_palette():
    print("==================================================")
    print(" 🎨 カラーパレット自動拡張ツール（Web標準140+色）")
    print("==================================================")

    # 既存のパレットを読み込み
    existing_palette = {}
    if os.path.exists(OUTPUT_JSON):
        try:
            with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
                existing_palette = json.load(f)
            print(f"📂 既存のパレットを読み込みました（登録数: {len(existing_palette)} 色）")
        except Exception as e:
            print(f"⚠️ パレット読み込みスキップ: {e}")

    css3_colors = get_css3_color_map()
    if not css3_colors:
        print("❌ エラー: webcolors からカラーマップを取得できませんでした。")
        return

    print(f"🔍 Web標準のカラーデータベースから全 {len(css3_colors)} 色を展開中...\n")

    added_count = 0
    updated_palette = existing_palette.copy()

    for color_name, hex_code in css3_colors.items():
        # キャメルケース風に見易く整形 (例: "darkblue" -> "Dark blue" または "Darkblue")
        formatted_name = color_name.replace("_", " ").title()
        
        # 既存パレットになければ追加
        if formatted_name not in updated_palette and color_name not in updated_palette:
            rgb = webcolors.hex_to_rgb(hex_code)
            bgr = [rgb.blue, rgb.green, rgb.red]
            
            updated_palette[formatted_name] = {
                "bgr": bgr,
                "label_jp": formatted_name,
                "en": color_name,
                "hex": hex_code,
                "dataset_count": 0
            }
            added_count += 1
            print(f"  ➕ 追加: {formatted_name} (HEX: {hex_code}, BGR: {bgr})")

    # 保存
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(updated_palette, f, indent=4, ensure_ascii=False)

    print("\n--------------------------------------------------")
    print(" ✨ 拡張完了！")
    print(f" ➕ 新しく追加された色: {added_count} 種類")
    print(f" 📦 最終的なカラーパレット総数: {len(updated_palette)} 種類")
    print(f" 💾 '{OUTPUT_JSON}' に保存しました。")
    print("--------------------------------------------------")

if __name__ == "__main__":
    expand_color_palette()