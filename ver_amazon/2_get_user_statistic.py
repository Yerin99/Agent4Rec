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
    # 사용자별 평가한 아이템 수 계산
    user_activity = ratings.groupby('user_id').size().reset_index(name='activity_num')
    
    # 활동성 점수 계산 (1-3)
    percentile = np.percentile(user_activity['activity_num'], [33, 66, 100])
    
    def activity_score(x):
        if x < percentile[0]:
            return 1  # 낮은 활동성
        elif x < percentile[1]:
            return 2  # 보통 활동성
        else:
            return 3  # 높은 활동성
    
    user_activity['activity'] = user_activity['activity_num'].apply(activity_score)
    
    print(f"활동성 점수 계산 완료. 각 레벨의 기준치: {percentile}")
    
    # 2. 다양성(diversity) 계산
    # 아이템과 평점 데이터 병합
    items_ratings = pd.merge(ratings, items, on='item_id', how='inner')
    
    # 카테고리 추출 및 집계
    category_counts = {}
    user_category_counts = {}
    
    for _, row in items_ratings.iterrows():
        user_id = row['user_id']
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
        
        # 사용자별 카테고리 카운트
        if user_id not in user_category_counts:
            user_category_counts[user_id] = {}
        
        for category in categories:
            # 전체 카테고리 카운트
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
            
            # 사용자별 카테고리 카운트
            if category in user_category_counts[user_id]:
                user_category_counts[user_id][category] += 1
            else:
                user_category_counts[user_id][category] = 1
    
    # 사용자별 고유 카테고리 수 계산
    diversity_data = []
    
    for user_id, categories in user_category_counts.items():
        diversity_num = len(categories)
        diversity_data.append({'user_id': user_id, 'diversity_num': diversity_num})
    
    diversity_df = pd.DataFrame(diversity_data)
    
    # 다양성 점수 계산 (1-3)
    percentile = np.percentile(diversity_df['diversity_num'], [33, 66, 100])
    
    def diversity_score(x):
        if x < percentile[0]:
            return 1  # 낮은 다양성
        elif x < percentile[1]:
            return 2  # 보통 다양성
        else:
            return 3  # 높은 다양성
    
    diversity_df['diversity'] = diversity_df['diversity_num'].apply(diversity_score)
    
    print(f"다양성 점수 계산 완료. 각 레벨의 기준치: {percentile}")
    
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
    
    # 순응성 점수 계산 (1-3)
    percentile = np.percentile(deviation['deviation'], [33, 66, 100])
    
    def conformity_score(x):
        if x < percentile[0]:
            return 3  # 높은 순응성 (낮은 편차)
        elif x < percentile[1]:
            return 2  # 보통 순응성
        else:
            return 1  # 낮은 순응성 (높은 편차)
    
    deviation['conformity'] = deviation['deviation'].apply(conformity_score)
    
    print(f"순응성 점수 계산 완료. 각 레벨의 기준치: {percentile}")
    
    # 모든 통계 병합
    statistics = pd.merge(user_activity[['user_id', 'activity']], 
                         diversity_df[['user_id', 'diversity']], 
                         on='user_id')
    
    statistics = pd.merge(statistics, 
                         deviation[['user_id', 'conformity']], 
                         on='user_id')
    
    statistics_num = pd.merge(user_activity[['user_id', 'activity_num']], 
                             diversity_df[['user_id', 'diversity_num']], 
                             on='user_id')
    
    statistics_num = pd.merge(statistics_num, 
                             deviation[['user_id', 'deviation']], 
                             on='user_id')
    
    # 결과 저장
    statistics.to_csv('data/simulation/user_statistic.csv', index=False)
    statistics_num.to_csv('data/simulation/user_statistic_num.csv', index=False)
    
    # 결과 출력
    print("\n사용자 통계 요약:")
    for _, row in statistics.iterrows():
        user_id = row['user_id']
        print(f"사용자 {user_id}:")
        print(f"  활동성(Activity): {row['activity']} (아이템 {statistics_num.loc[statistics_num['user_id'] == user_id, 'activity_num'].values[0]}개)")
        print(f"  다양성(Diversity): {row['diversity']} (카테고리 {statistics_num.loc[statistics_num['user_id'] == user_id, 'diversity_num'].values[0]}개)")
        print(f"  순응성(Conformity): {row['conformity']} (평균 평점과의 차이: {statistics_num.loc[statistics_num['user_id'] == user_id, 'deviation'].values[0]:.2f})")
    
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