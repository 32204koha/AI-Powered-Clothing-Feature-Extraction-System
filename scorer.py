from collections import Counter
from typing import Dict, Any, List


class DeclutterScorer:
    """
    断捨離支援AI ルールベース・スコア算出クラス
    （色・カテゴリーグループ化、同一服の重複除外、類似服の追加スコア加算に対応）
    """

    def __init__(self, wardrobe_items: List[Dict[str, Any]]):
        """
        クローゼット全体の衣類データを受け取り、色とカテゴリーの所持数を集計する。
        ※「同じ服（is_same_item=True）」として登録されたアイテムは重複カウントの母数から除外する。
        """
        unique_items = [
            item for item in wardrobe_items
            if not item.get("is_same_item", False)
        ]
        self.color_counts = Counter(item["color"] for item in unique_items)
        self.category_counts = Counter(item["category"] for item in unique_items)

    def check_duplicates(self, color: str, category: str, is_same_item: bool = False) -> Dict[str, bool]:
        """
        Q2: 自動取得データから同じ色・カテゴリーが3着以上あるかを判定
        ※同一服として登録されたアイテム自身は重複カウントしない
        """
        if is_same_item:
            return {
                "has_duplicate_color": False,
                "has_duplicate_category": False,
            }
        return {
            "has_duplicate_color": self.color_counts[color] >= 3,
            "has_duplicate_category": self.category_counts[category] >= 3,
        }

    def get_required_questions(self, q1_answer: str, dup_status: Dict[str, bool]) -> List[str]:
        """
        条件分岐に基づき、ユーザーに追加で出題すべき質問（Q3, Q4）を判定する
        """
        questions = ["Q1", "Q5"]

        if q1_answer in ["1年以内", "覚えていない"]:
            questions.append("Q3")

        is_dup = dup_status["has_duplicate_color"] or dup_status["has_duplicate_category"]
        is_long_unused = q1_answer in ["半年以内", "1年以内", "覚えていない"]
        if is_dup or is_long_unused:
            questions.append("Q4")

        return sorted(questions)

    @staticmethod
    def get_judgment_label(score_percent: float) -> str:
        """
        最終スコア（%）から3段階の判定ラベルを返す
        """
        if score_percent >= 70.0:
            return "断捨離推奨"
        elif score_percent >= 50.0:
            return "要検討"
        else:
            return "キープ"

    def calculate_score(self, item_info: Dict[str, Any], answers: Dict[str, str]) -> Dict[str, Any]:
        """
        最終スコア（%）、判定ラベル、内訳を算出する（動的正規化）
        """
        total_score = 0
        max_possible_score = 0
        breakdown = {}

        is_same_item = item_info.get("is_same_item", False)
        has_similar_candidate = item_info.get("has_similar_candidate", False)

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
        dup_status = self.check_duplicates(
            item_info["color"],
            item_info["category"],
            is_same_item=is_same_item,
        )
        q2_pt = 0
        if dup_status["has_duplicate_color"]:
            q2_pt += 15
        if dup_status["has_duplicate_category"]:
            q2_pt += 15

        total_score += q2_pt
        max_possible_score += 30
        if is_same_item:
            breakdown["Q2_重複状況"] = "0/30点 (同一服登録のため重複カウント除外)"
        else:
            breakdown["Q2_重複状況"] = (
                f"{q2_pt}/30点 (同色3着以上:{'あり' if dup_status['has_duplicate_color'] else 'なし'}, "
                f"同カテゴリ3着以上:{'あり' if dup_status['has_duplicate_category'] else 'なし'})"
            )

        # ---------------------------------------------------------
        # 2.5 視覚的類似ボーナス（似た服があるが「別の服」と回答した場合）
        # ---------------------------------------------------------
        if has_similar_candidate and not is_same_item:
            total_score += 15
            max_possible_score += 15
            breakdown["Q_視覚的類似"] = "15/15点 (似たデザインの別の服を所持)"
        elif has_similar_candidate and is_same_item:
            breakdown["Q_視覚的類似"] = "スキップ (既登録の同一服として紐づけ)"

        # ---------------------------------------------------------
        # 3. 見て心が動くか（思い出）【条件分岐】
        # ---------------------------------------------------------
        if q1_ans in ["1年以内", "覚えていない"] and "Q3" in answers:
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

        if (is_dup or is_long_unused) and "Q4" in answers:
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

        # 最終スコア（パーセンテージ）と判定ラベルの算出
        final_percentage = round((total_score / max_possible_score) * 100, 1)
        judgment = self.get_judgment_label(final_percentage)

        return {
            "item_id": item_info["id"],
            "total_score": total_score,
            "max_possible_score": max_possible_score,
            "final_score_percent": final_percentage,
            "judgment": judgment,
            "breakdown": breakdown,
        }

    @classmethod
    def recalculate_all(cls, wardrobe_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        登録順の優劣をなくすため、現在のクローゼット全体を使って全アイテムのスコアを再計算する
        """
        if not wardrobe_items:
            return []

        scorer = cls(wardrobe_items)
        updated_items = []

        for item in wardrobe_items:
            score_res = scorer.calculate_score(item, item["answers"])
            item["total_score"] = score_res["total_score"]
            item["max_possible_score"] = score_res["max_possible_score"]
            item["final_score_percent"] = score_res["final_score_percent"]
            item["judgment"] = score_res["judgment"]
            item["breakdown"] = score_res["breakdown"]
            updated_items.append(item)

        return updated_items

    @staticmethod
    def summarize_wardrobe(wardrobe_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        クローゼット全体の集計（平均スコア、判定別件数）を算出する
        """
        total_count = len(wardrobe_items)
        if total_count == 0:
            return {
                "total_count": 0,
                "average_score": 0.0,
                "counts": {
                    "断捨離推奨": 0,
                    "要検討": 0,
                    "キープ": 0,
                },
            }

        scores = [item["final_score_percent"] for item in wardrobe_items]
        avg_score = round(sum(scores) / total_count, 1)

        counts = {
            "断捨離推奨": sum(1 for item in wardrobe_items if item["judgment"] == "断捨離推奨"),
            "要検討": sum(1 for item in wardrobe_items if item["judgment"] == "要検討"),
            "キープ": sum(1 for item in wardrobe_items if item["judgment"] == "キープ"),
        }

        return {
            "total_count": total_count,
            "average_score": avg_score,
            "counts": counts,
        }