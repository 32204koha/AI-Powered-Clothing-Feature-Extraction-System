# ==========================================
# 服の断捨離スコアを計算するモジュール
# ==========================================

def calculate_item_score(attributes: dict, survey_data: str) -> int:
    """1着ごとの服のスコアを計算する"""
    score = 50 
    
    if isinstance(attributes, dict):
        if attributes.get("gender") == "Men":
            score += 10
            
    if survey_data == "好き":
        score += 30
    elif survey_data == "普通":
        score += 10
        
    return score


def calculate_total_score(closet_data: list) -> int:
    """クローゼット全体の健康度スコア（100点満点）を計算する"""
    total_items = len(closet_data)
    if total_items == 0:
        return 100
        
    score = 100
    for item in closet_data:
        # アンケートによる減点
        if item.get("survey_result") == "嫌い":
            score -= 20
        elif item.get("survey_result") == "保留":
            score -= 10
            
        # 状態による減点
        if item.get("condition") == "使用感あり":
            score -= 5
            
    return max(0, min(100, score))