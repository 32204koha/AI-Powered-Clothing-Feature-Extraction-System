import itertools
from collections import Counter
from score import DeclutterScorer

# ※先ほどの DeclutterScorer クラスが定義されている前提の検証コードです

def run_exhaustive_simulation():
    # 1. 重複あり（色3着・カテゴリ3着）と重複なしの2パターンのクローゼットを用意
    wardrobe_dup = [
        {"id": "test", "color": "黒", "category": "トップス"},
        {"id": "d1", "color": "黒", "category": "パンツ"},
        {"id": "d2", "color": "黒", "category": "アウター"},
        {"id": "d3", "color": "白", "category": "トップス"},
        {"id": "d4", "color": "青", "category": "トップス"},
    ]
    wardrobe_no_dup = [
        {"id": "test", "color": "赤", "category": "ワンピース"},
    ]

    q1_choices = ["1週間以内", "1か月以内", "半年以内", "1年以内", "覚えていない"]
    q3_choices = ["はい", "なんとなく", "思い出せない"]
    q4_choices = ["はい", "迷う", "いいえ"]
    q5_choices = ["何通りか想像できる", "いつも決まったパターンで着ている", "着用イメージが湧かない"]

    results = []
    anomalies = []  # 論理的矛盾（異常値）を記録

    for dup_mode, wardrobe in [("重複MAX", wardrobe_dup), ("重複なし", wardrobe_no_dup)]:
        scorer = DeclutterScorer(wardrobe)
        item = wardrobe[0]

        for q1, q3, q4, q5 in itertools.product(q1_choices, q3_choices, q4_choices, q5_choices):
            answers = {"Q1": q1, "Q3": q3, "Q4": q4, "Q5": q5}
            res = scorer.calculate_score(item, answers)
            score = res["final_score_percent"]
            results.append((score, dup_mode, q1, q3, q4, q5, res["breakdown"]))

            # 【異常値チェック例】「1週間以内」に着た服なのにスコアが50%を超えているケース
            if q1 == "1週間以内" and score >= 50.0:
                anomalies.append((score, dup_mode, answers))

    # 重複ループ分のユニーク化（スキップされた質問の違いを統合）
    unique_results = list({(r[0], r[1], str(r[6])): r for r in results}.values())
    unique_results.sort(key=lambda x: x[0], reverse=True)

    print(f"=== シミュレーション総パターン数（ユニーク）: {len(unique_results)}通り ===\n")

    # 1. スコア分布の表示
    bins = {"0-20%": 0, "21-40%": 0, "41-60%": 0, "61-80%": 0, "81-100%": 0}
    for r in unique_results:
        s = r[0]
        if s <= 20: bins["0-20%"] += 1
        elif s <= 40: bins["21-40%"] += 1
        elif s <= 60: bins["41-60%"] += 1
        elif s <= 80: bins["61-80%"] += 1
        else: bins["81-100%"] += 1

    print("【スコア分布（ヒストグラム）】")
    for k, v in bins.items():
        print(f"  {k}: {v}パターン ({'#' * v})")

    # 2. 異常値（逆転現象）の報告
    print(f"\n【論理的矛盾（1週間以内着用で50%以上）の検出数】: {len(anomalies)}件")
    for a in set((x[0], x[1], str(x[2])) for x in anomalies):
        print(f"  スコア {a[0]}% | {a[1]} | 回答: {a[2]}")

    # 3. 境界線（グレーゾーン：60%〜75%付近）のパターン確認
    print("\n【しきい値付近（60%〜75%）の判定例】")
    for r in unique_results:
        if 60.0 <= r[0] <= 75.0:
            print(f"  {r[0]}% | {r[1]} | Q1:{r[2]}, Q3:{r[3]}, Q4:{r[4]}, Q5:{r[5]}")

if __name__ == "__main__":
    run_exhaustive_simulation()