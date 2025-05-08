import numpy as np
import pandas as pd
import os
import random
import torch
import pickle
import json

def fix_seeds(seed=101):
    """
    코드 실행 결과의 재현성을 위해 랜덤 시드를 고정합니다.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def calculate_user_statistics():
    """
    사용자 통계를 계산합니다.
    - 활동성(activity): 사용자가 평가한 아이템 수
    - 다양성(diversity): 사용자가 평가한 아이템의 카테고리 다양성
    - 순응성(conformity): 사용자의 평점이 평균 평점과 얼마나 일치하는지
    """
    print("사용자 통계 계산 중...")
    
    # 필요한 디렉토리 생성
    os.makedirs('data/simulation', exist_ok=True)
    
    # 데이터 로드
    ratings = pd.read_csv('data/cf_data/all_ratings.csv')
    items = pd.read_csv('data/cf_data/all_items.csv')
    
    # 1. 활동성(activity) 계산
    user_activity = ratings.groupby('user_id').size().reset_index(name='activity_num')
    
    # 단일 사용자이므로 활동성 등급을 2(중간)로 설정
    user_activity['activity'] = 2
    
    # 2. 다양성(diversity) 계산
    # 아이템과 평점 데이터 병합
    items_ratings = pd.merge(ratings, items, on='item_id', how='inner')
    
    # 카테고리 추출 및 집계
    category_counts = {}
    
    for _, row in items_ratings.iterrows():
        categories = row['category']
        if isinstance(categories, str):
            try:
                # 문자열로 된 리스트를 실제 리스트로 변환
                # Amazon 데이터 형식에 맞게 처리
                if categories.startswith('[') and categories.endswith(']'):
                    categories = json.loads(categories.replace("'", '"'))
                else:
                    categories = [categories]
            except json.JSONDecodeError:
                # JSON 파싱 오류 처리
                categories = [categories]
        elif isinstance(categories, list):
            # 이미 리스트 형태인 경우
            pass
        elif categories is None:
            categories = []
        else:
            # 다른 형태의 데이터는 문자열로 변환하여 단일 카테고리로 처리
            categories = [str(categories)]
            
        # 각 카테고리 카운트
        for category in categories:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
    
    # 카테고리 다양성 계산 (고유 카테고리 수)
    diversity_num = len(category_counts)
    
    # 다양성 등급 설정 (1-3)
    if diversity_num <= 3:
        diversity = 1
    elif diversity_num <= 7:
        diversity = 2
    else:
        diversity = 3
    
    # 3. 순응성(conformity) 계산
    # 각 아이템의 평균 평점
    avg_ratings = ratings.groupby('item_id')['overall'].mean().reset_index()
    avg_ratings.rename(columns={'overall': 'avg_rating'}, inplace=True)
    
    # 사용자 평점과 평균 평점 비교
    ratings_with_avg = pd.merge(ratings, avg_ratings, on='item_id')
    ratings_with_avg['mse'] = (ratings_with_avg['overall'] - ratings_with_avg['avg_rating']) ** 2
    
    # 평균 제곱 오차(MSE) 계산
    deviation = ratings_with_avg.groupby('user_id')['mse'].mean().reset_index()
    deviation.rename(columns={'mse': 'deviation'}, inplace=True)
    
    # 순응성 등급 설정 (1-3)
    if deviation['deviation'].iloc[0] < 0.5:
        conformity = 1  # 높은 순응성
    elif deviation['deviation'].iloc[0] < 1.5:
        conformity = 2  # 중간 순응성
    else:
        conformity = 3  # 낮은 순응성
    
    # 모든 통계 병합
    statistics = pd.DataFrame({
        'user_id': [0],  # 단일 사용자
        'activity': [user_activity['activity'].iloc[0]],
        'diversity': [diversity],
        'conformity': [conformity]
    })
    
    statistics_num = pd.DataFrame({
        'user_id': [0],  # 단일 사용자
        'activity_num': [user_activity['activity_num'].iloc[0]],
        'diversity_num': [diversity_num],
        'deviation': [deviation['deviation'].iloc[0]]
    })
    
    # 결과 저장
    statistics.to_csv('data/simulation/user_statistic.csv', index=False)
    statistics_num.to_csv('data/simulation/user_statistic_num.csv', index=False)
    
    print("사용자 통계:")
    print(f"활동성(Activity): {user_activity['activity'].iloc[0]} (아이템 {user_activity['activity_num'].iloc[0]}개)")
    print(f"다양성(Diversity): {diversity} (카테고리 {diversity_num}개)")
    print(f"순응성(Conformity): {conformity} (평균 평점과의 차이: {deviation['deviation'].iloc[0]:.2f})")
    
    # 카테고리 정보 출력
    print("\n가장 많이 평가된 카테고리 Top 5:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"- {category}: {count}개 아이템")
    
    return statistics, statistics_num

def main():
    seed = 101
    fix_seeds(seed)
    
    # 사용자 통계 계산
    statistics, statistics_num = calculate_user_statistics()
    
    print("사용자 통계 분석 완료!")

if __name__ == "__main__":
    main() 