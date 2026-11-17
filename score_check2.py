import random
from typing import List, Dict, Tuple

# ※ご自身の環境に合わせて、DeclutterScorerクラスを定義したファイル名に変更してください
from score import DeclutterScorer


def generate_persona_wardrobe(persona_type: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    ペルソナの特性に応じた仮想クローゼット（衣類データ）と回答データを生成する
    """
    colors = ["黒", "白", "グレー", "ネイビー", "ベージュ", "赤", "青", "緑"]
    categories = ["トップス", "パンツ", "アウター", "ワンピース", "スカート"]

    q1_choices = ["1週間以内", "1か月以内", "半年以内", "1年以内", "覚えていない"]
    q3_choices = ["はい", "なんとなく", "思い出せない"]
    q4_choices = ["はい", "迷う", "いいえ"]
    q5_choices = ["何通りか想像できる", "いつも決まったパターンで着ている", "着用イメージが湧かない"]

    # ペルソナごとのパラメータ設定（所持数、色・カテゴリの偏り、回答の確率重み）
    if persona_type == "溜め込み型（マキシマリスト）":
        num_items = 60
        # 黒・白・グレーとトップスに極端に偏る（重複が多発する）
        color_weights = [30, 25, 20, 10, 5, 3, 4, 3]
        cat_weights = [50, 20, 15, 5, 10]
        # 未着用期間が長く、思い出せない・もう買わない傾向が強い
        q1_w = [5, 10, 20, 30, 35]
        q3_w = [15, 35, 50]
        q4_w = [10, 30, 60]
        q5_w = [20, 30, 50]

    elif persona_type == "厳選型（ミニマリスト）":
        num_items = 15
        # 色・カテゴリが分散しており重複が少ない
        color_weights = [15, 15, 15, 15, 15, 10, 10, 5]
        cat_weights = [25, 25, 20, 15, 15]
        # 直近で着用しており、再購入意欲や着用イメージが高い
        q1_w = [50, 35, 10, 5, 0]
        q3_w = [70, 25, 5]
        q4_w = [70, 25, 5]
        q5_w = [50, 45, 5]

    else:  # "一般層（迷いやすい平均的ユーザー）"
        num_items = 35
        # ベーシックカラー中心にやや偏りあり
        color_weights = [25, 20, 15, 15, 10, 5, 5, 5]
        cat_weights = [35, 25, 20, 10, 10]
        # 一軍（直近）と二軍・三軍（長期放置）が混在
        q1_w = [20, 25, 25, 15, 15]
        q3_w = [30, 45, 25]
        q4_w = [30, 40, 30]
        q5_w = [35, 40, 25]

    wardrobe = []
    answers_list = []

    for i in range(1, num_items + 1):
        item = {
            "id": f"item_{i:02d}",
            "color": random.choices(colors, weights=color_weights, k=1)[0],
            "category": random.choices(categories, weights=cat_weights, k=1)[0],
        }
        wardrobe.append(item)

        ans = {
            "Q1": random.choices(q1_choices, weights=q1_w, k=1)[0],
            "Q3": random.choices(q3_choices, weights=q3_w, k=1)[0],
            "Q4": random.choices(q4_choices, weights=q4_w, k=1)[0],
            "Q5": random.choices(q5_choices, weights=q5_w, k=1)[0],
        }
        answers_list.append(ans)

    return wardrobe, answers_list


def run_persona_simulation():
    # 論文記載用に結果を固定（再現性の確保）
    random.seed(42)

    personas = [
        "溜め込み型（マキシマリスト）",
        "一般層（迷いやすい平均的ユーザー）",
        "厳選型（ミニマリスト）",
    ]

    print("=================================================================")
    print(" ペルソナ別・仮想クローゼット スコアリング妥当性シミュレーション")
    print("=================================================================\n")

    for persona in personas:
        wardrobe, answers_list = generate_persona_wardrobe(persona)
        scorer = DeclutterScorer(wardrobe)

        scores = []
        recommend_discard = 0  # 70%以上：断捨離推奨
        consider = 0           # 50〜69.9%：要検討
        keep = 0               # 49.9%以下：キープ

        for item, ans in zip(wardrobe, answers_list):
            res = scorer.calculate_score(item, ans)
            s = res["final_score_percent"]
            scores.append(s)

            if s >= 70.0:
                recommend_discard += 1
            elif s >= 50.0:
                consider += 1
            else:
                keep += 1

        total = len(wardrobe)
        avg_score = round(sum(scores) / total, 1)

        print(f"【ペルソナ：{persona}】（総所持数：{total}着）")
        print(f"  平均断捨離スコア : {avg_score}% （最小: {min(scores)}% / 最大: {max(scores)}%）")
        print(f"  ・断捨離推奨 (70%〜)   : {recommend_discard:2d}着 ({recommend_discard / total * 100:5.1f}%)")
        print(f"  ・要検討     (50〜69%) : {consider:2d}着 ({consider / total * 100:5.1f}%)")
        print(f"  ・キープ     (〜49%)   : {keep:2d}着 ({keep / total * 100:5.1f}%)")
        print("-" * 65)


if __name__ == "__main__":
    run_persona_simulation()