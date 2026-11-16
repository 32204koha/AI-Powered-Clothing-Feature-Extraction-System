from collections import Counter
from typing import Dict, Any, List


class DeclutterScorer:
    """
    断捨離支援AI ルールベース・スコア算出クラス
    """

    def __init__(self, wardrobe_items: List[Dict[str, str]]):
        """
        クローゼット全体の衣類データを受け取り、色とカテゴリーの所持数を集計する
        :param wardrobe_items: [{"id": "item_1", "color": "黒", "category": "トップス"}, ...]
        """
        self.color_counts = Counter(item["color"] for item in wardrobe_items)
        self.category_counts = Counter(item["category"] for item in wardrobe_items)

    def check_duplicates(self, color: str, category: str) -> Dict[str, bool]:
        """
        Q2: 自動取得データから同じ色・カテゴリーが3着以上あるかを判定
        """
        return {
            "has_duplicate_color": self.color_counts[color] >= 3,
            "has_duplicate_category": self.category_counts[category] >= 3,
        }

    def get_required_questions(self, q1_answer: str, dup_status: Dict[str, bool]) -> List[str]:
        """
        条件分岐に基づき、ユーザーに追加で出題すべき質問（Q3, Q4）を判定する
        """
        questions = ["Q1", "Q5"]  # Q1とQ5は必須

        # Q3の発生条件: Q1で「1年以内」または「覚えていない」を選択
        if q1_answer in ["1年以内", "覚えていない"]:
            questions.append("Q3")

        # Q4の発生条件: 同じ色・カテゴリーが3着以上 または Q1が「半年以内」「1年以内」「覚えていない」
        is_dup = dup_status["has_duplicate_color"] or dup_status["has_duplicate_category"]
        is_long_unused = q1_answer in ["半年以内", "1年以内", "覚えていない"]
        if is_dup or is_long_unused:
            questions.append("Q4")

        return sorted(questions)

    def calculate_score(self, item_info: Dict[str, str], answers: Dict[str, str]) -> Dict[str, Any]:
        """
        最終スコア（%）と内訳を算出する
        """
        total_score = 0
        max_possible_score = 0
        breakdown = {}

        # ---------------------------------------------------------
        # 1. 一年使っていない（時間軸）【必須】
        # ---------------------------------------------------------
        q1_weights = {
            "1週間以内": 0,
            "1か月以内": 10,
            "半年以内": 20,
            "1年以内": 30,
            "覚えていない": 50,
        }
        q1_ans = answers["Q1"]
        q1_pt = q1_weights[q1_ans]
        total_score += q1_pt
        max_possible_score += 50
        breakdown["Q1_時間軸"] = f"{q1_pt}/50点 ({q1_ans})"

        # ---------------------------------------------------------
        # 2. 代用品があるか（自動取得）【必須】
        # ---------------------------------------------------------
        dup_status = self.check_duplicates(item_info["color"], item_info["category"])
        q2_pt = 0
        if dup_status["has_duplicate_color"]:
            q2_pt += 15
        if dup_status["has_duplicate_category"]:
            q2_pt += 15

        total_score += q2_pt
        max_possible_score += 30  # 色(15点) + カテゴリー(15点)
        breakdown["Q2_重複状況"] = f"{q2_pt}/30点 (色重複:{dup_status['has_duplicate_color']}, カテゴリ重複:{dup_status['has_duplicate_category']})"

        # ---------------------------------------------------------
        # 3. 見て心が動くか（思い出）【条件分岐】
        # ---------------------------------------------------------
        if q1_ans in ["1年以内", "覚えていない"]:
            q3_weights = {
                "はい": 0,
                "なんとなく": 10,
                "思い出せない": 30,
            }
            q3_ans = answers["Q3"]
            q3_pt = q3_weights[q3_ans]
            total_score += q3_pt
            max_possible_score += 30
            breakdown["Q3_思い出"] = f"{q3_pt}/30点 ({q3_ans})"
        else:
            breakdown["Q3_思い出"] = "スキップ"

        # ---------------------------------------------------------
        # 4. 今持っていなかったらもう一度買うか【条件分岐】
        # ---------------------------------------------------------
        is_dup = dup_status["has_duplicate_color"] or dup_status["has_duplicate_category"]
        is_long_unused = q1_ans in ["半年以内", "1年以内", "覚えていない"]

        if is_dup or is_long_unused:
            q4_weights = {
                "はい": 0,
                "迷う": 10,
                "いいえ": 30,
            }
            q4_ans = answers["Q4"]
            q4_pt = q4_weights[q4_ans]
            total_score += q4_pt
            max_possible_score += 30
            breakdown["Q4_再購入意欲"] = f"{q4_pt}/30点 ({q4_ans})"
        else:
            breakdown["Q4_再購入意欲"] = "スキップ"

        # ---------------------------------------------------------
        # 5. この服を着ている自分を想像できるか【必須】
        # ---------------------------------------------------------
        q5_weights = {
            "何通りか想像できる": 0,
            "いつも決まったパターンで着ている": 0,
            "着用イメージが湧かない": 10,
        }
        q5_ans = answers["Q5"]
        q5_pt = q5_weights[q5_ans]
        total_score += q5_pt
        max_possible_score += 10
        breakdown["Q5_着用イメージ"] = f"{q5_pt}/10点 ({q5_ans})"

        # ---------------------------------------------------------
        # 6. （今後追加する場合はここに同様のブロックを記述）
        # ---------------------------------------------------------

        # 最終スコア（パーセンテージ）の算出
        final_percentage = round((total_score / max_possible_score) * 100, 1)

        return {
            "item_id": item_info["id"],
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "final_score_percent": final_percentage,
            "breakdown": breakdown,
        }


# =========================================================
# 動作確認用シミュレーション
# =========================================================
if __name__ == "__main__":
    # 仮想のクローゼットデータ（黒が3着、トップスが3着ある状態）
    my_wardrobe = [
        {"id": "item_01", "color": "黒", "category": "トップス"},
        {"id": "item_02", "color": "黒", "category": "パンツ"},
        {"id": "item_03", "color": "黒", "category": "アウター"},
        {"id": "item_04", "color": "白", "category": "トップス"},
        {"id": "item_05", "color": "青", "category": "トップス"},
    ]

    scorer = DeclutterScorer(my_wardrobe)

    # ケース1：重複あり（黒3着・トップス3着）＋ 1年以上未着用（全質問発生）
    ans_case1 = {
        "Q1": "覚えていない",
        "Q3": "思い出せない",
        "Q4": "いいえ",
        "Q5": "着用イメージが湧かない",
    }
    res1 = scorer.calculate_score(my_wardrobe[0], ans_case1)

    # ケース2：重複なし（青1着・トップス3着）＋ 1週間以内に着用（Q3はスキップ、Q4はカテゴリ重複で発生）
    ans_case2 = {
        "Q1": "1週間以内",
        "Q4": "はい",
        "Q5": "いつも決まったパターンで着ている",
    }
    res2 = scorer.calculate_score(my_wardrobe[4], ans_case2)

    for res in [res1, res2]:
        print(f"--- アイテムID: {res['item_id']} ---")
        print(f"最終断捨離スコア: {res['final_score_percent']}% ({res['total_score']}点 / 最大{res['max_possible_score']}点)")
        for k, v in res["breakdown"].items():
            print(f"  {k}: {v}")
        print()