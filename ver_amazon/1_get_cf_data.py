import numpy as np
import pandas as pd
import os
import random
import json
import torch
import pickle

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

def load_data():
    """
    Amazon Movies and TV 데이터셋을 로드합니다.
    """
    print("데이터 로드 중...")
    
    # 파일 경로 설정
    reviews_path = 'data/Movies_and_TV.json'
    meta_path = 'data/meta_Movies_and_TV.json'
    
    # JSON 파일을 리스트로 로드
    reviews = []
    count = 0
    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                reviews.append(json.loads(line.strip()))
                count += 1
                if count >= 100000:  # 데이터 크기 제한
                    break
            except json.JSONDecodeError:
                continue
    
    # 메타데이터 로드
    meta = []
    count = 0
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                meta.append(json.loads(line.strip()))
                count += 1
                if count >= 50000:  # 데이터 크기 제한
                    break
            except json.JSONDecodeError:
                continue
    
    print(f"로드된 리뷰 수: {len(reviews)}")
    print(f"로드된 메타데이터 수: {len(meta)}")
    
    return reviews, meta

def int_to_user_dict(interaction):
    """
    상호작용 리스트를 사용자 딕셔너리로 변환합니다.
    입력: ['user_id', 'item_id'] 컬럼이 있는 DataFrame
    출력: key: user_id, value: item_id 리스트인 딕셔너리
    """
    user_dict = {}
    for u, v in interaction:
        if u not in user_dict.keys():
            user_dict[u] = [v]
        else:
            user_dict[u].append(v)
    # 키 기준 정렬
    user_dict = dict(sorted(user_dict.items(), key=lambda x: x[0]))
    return user_dict

def save_user_dict_to_txt(user_dict, filepath):
    """
    사용자 딕셔너리를 텍스트 파일로 저장합니다.
    """
    with open(filepath, 'w') as f:
        for u, items in user_dict.items():
            f.write(str(int(u)))
            for i in items:
                f.write(' ' + str(int(i)))
            f.write('\n')

def create_cf_data(reviews, meta):
    """
    협업 필터링을 위한 데이터를 준비합니다.
    """
    # 리뷰 데이터를 데이터프레임으로 변환
    reviews_df = pd.DataFrame(reviews)
    
    # ASIN을 문자열로 명시적 변환
    reviews_df['asin'] = reviews_df['asin'].astype(str)
    
    # 메타 데이터를 데이터프레임으로 변환
    meta_df = pd.DataFrame(meta)
    
    # ASIN을 문자열로 명시적 변환
    if 'asin' in meta_df.columns:
        meta_df['asin'] = meta_df['asin'].astype(str)
    
    # 활동량이 많은 사용자 선택 (상위 10명)
    reviewer_counts = reviews_df['reviewerID'].value_counts().head(10)
    top_reviewer_ids = reviewer_counts.index.tolist()
    
    print(f"선택된 상위 10명의 사용자 ID: {top_reviewer_ids}")
    print(f"각 사용자의 리뷰 수: {reviewer_counts.values}")
    
    # 선택된 사용자의 리뷰만 필터링
    filtered_reviews = reviews_df[reviews_df['reviewerID'].isin(top_reviewer_ids)]
    
    # 사용자 ID와 아이템 ID 매핑 (정수로 변환)
    user_id_map = {user_id: idx for idx, user_id in enumerate(top_reviewer_ids)}
    
    # 해당 사용자가 리뷰한 아이템들만 선택
    user_items = filtered_reviews['asin'].unique()
    item_id_map = {asin: idx for idx, asin in enumerate(user_items)}
    
    # 선택된 아이템의 메타데이터만 필터링
    filtered_meta = meta_df[meta_df['asin'].isin(user_items)]
    
    # 필요한 디렉토리 생성
    os.makedirs('data/cf_data', exist_ok=True)
    
    # 매핑 정보 저장
    with open('data/cf_data/user_id_map.pkl', 'wb') as f:
        pickle.dump(user_id_map, f)
    
    with open('data/cf_data/item_id_map.pkl', 'wb') as f:
        pickle.dump(item_id_map, f)
    
    # 매핑된 ID로 리뷰 데이터 변환
    filtered_reviews['user_id'] = filtered_reviews['reviewerID'].map(user_id_map)
    filtered_reviews['item_id'] = filtered_reviews['asin'].map(item_id_map)
    
    # 긍정적인 평가만 필터링 (평점 > 3)
    positive_reviews = filtered_reviews[filtered_reviews['overall'] > 3]
    
    # 훈련/검증/테스트 세트 분할 (6:2:2)
    train_pairs = positive_reviews.sample(frac=0.6, random_state=101)
    temp = positive_reviews[~positive_reviews.index.isin(train_pairs.index)]
    valid_pairs = temp.sample(frac=0.5, random_state=101)
    test_pairs = temp[~temp.index.isin(valid_pairs.index)]
    
    # 사용자-아이템 상호작용 저장
    train_interactions = train_pairs[['user_id', 'item_id']].values.tolist()
    valid_interactions = valid_pairs[['user_id', 'item_id']].values.tolist()
    test_interactions = test_pairs[['user_id', 'item_id']].values.tolist()
    
    # 사용자 별 아이템 목록 생성
    train_user_dict = int_to_user_dict(train_interactions)
    valid_user_dict = int_to_user_dict(valid_interactions)
    test_user_dict = int_to_user_dict(test_interactions)
    
    # 사용자-아이템 상호작용을 텍스트 파일로 저장
    save_user_dict_to_txt(train_user_dict, 'data/cf_data/train.txt')
    save_user_dict_to_txt(valid_user_dict, 'data/cf_data/valid.txt')
    save_user_dict_to_txt(test_user_dict, 'data/cf_data/test.txt')
    
    # 평점별 아이템 분류
    rating_groups = filtered_reviews.groupby('overall')
    
    rating_item_lists = {}
    for rating, group in rating_groups:
        rating_item_lists[rating] = group['asin'].tolist()
    
    # 평점별 아이템 목록 저장
    with open('data/cf_data/rating_item_lists.pkl', 'wb') as f:
        pickle.dump(rating_item_lists, f)
    
    # 모든 평점 데이터 저장 (페르소나 생성에 사용)
    # ASIN을 문자열로 유지하기 위해 dtypes 지정
    filtered_reviews[['user_id', 'item_id', 'overall', 'reviewText', 'summary', 'asin']].to_csv(
        'data/cf_data/all_ratings.csv', index=False)
    
    # 아이템 정보 저장
    if 'title' in filtered_meta.columns:
        item_columns = ['asin', 'title', 'main_cat', 'category']
    else:
        # title 컬럼이 없는 경우 다른 컬럼으로 대체
        filtered_meta['title'] = filtered_meta.get('name', filtered_meta.get('description', 'Unknown'))
        item_columns = ['asin', 'title', 'main_cat', 'category']
    
    item_df = filtered_meta[item_columns].copy() if all(col in filtered_meta.columns for col in item_columns) else filtered_meta.copy()
    
    # 필요한 컬럼이 없으면 추가
    for col in item_columns:
        if col not in item_df.columns:
            item_df[col] = 'Unknown'
    
    # ASIN 문자열 타입 유지
    item_df['asin'] = item_df['asin'].astype(str)
    item_df['item_id'] = item_df['asin'].map(item_id_map)
    
    # item_id가 없는 행은 제외
    item_df = item_df.dropna(subset=['item_id'])
    
    # 카테고리가 리스트인 경우 문자열로 변환
    if 'category' in item_df.columns:
        item_df['category'] = item_df['category'].apply(lambda x: str(x) if isinstance(x, list) else x)
    
    # 아이템 정보 저장
    item_df.to_csv('data/cf_data/all_items.csv', index=False)
    
    print("협업 필터링 데이터 준비 완료!")
    print(f"사용자 수: {len(user_id_map)}")
    print(f"아이템 수: {len(item_id_map)}")
    print(f"훈련 상호작용 수: {len(train_interactions)}")
    print(f"검증 상호작용 수: {len(valid_interactions)}")
    print(f"테스트 상호작용 수: {len(test_interactions)}")
    
    return filtered_reviews, filtered_meta, rating_item_lists

def main():
    seed = 101
    fix_seeds(seed)
    
    print("Amazon Movies and TV 데이터 로드 중...")
    reviews, meta = load_data()
    
    print("협업 필터링 데이터 준비 중...")
    filtered_reviews, filtered_meta, rating_item_lists = create_cf_data(reviews, meta)
    
    print("초기 프로필 데이터 생성 중...")
    # 상위 25개 항목으로 초기 프로필 생성
    n_for_init = 25
    
    # 사용자의 평점별 아이템 목록 생성
    user_profiles = []
    
    for user_id in range(len(set(filtered_reviews['user_id']))):
        user_profile = {}
        user_reviews = filtered_reviews[filtered_reviews['user_id'] == user_id]
        
        for rating in range(1, 6):
            items_with_rating = user_reviews[user_reviews['overall'] == rating]
            if not items_with_rating.empty:
                # 아이템 제목 가져오기
                items_with_details = pd.merge(
                    items_with_rating, 
                    filtered_meta, 
                    on='asin', 
                    how='inner'
                )
                
                if not items_with_details.empty and 'title' in items_with_details.columns:
                    titles = items_with_details['title'].tolist()
                    user_profile[rating] = "; ".join(titles[:n_for_init])
                else:
                    user_profile[rating] = "None"
            else:
                user_profile[rating] = "None"
        
        user_profile['user_id'] = user_id
        user_profiles.append(user_profile)
    
    # 초기 프로필 저장
    profile_df = pd.DataFrame(user_profiles)
    
    # 아이템 ID도 함께 저장
    for user_id in range(len(set(filtered_reviews['user_id']))):
        user_reviews = filtered_reviews[filtered_reviews['user_id'] == user_id]
        
        for rating in range(1, 6):
            items_with_rating = user_reviews[user_reviews['overall'] == rating]
            if not items_with_rating.empty:
                profile_df.loc[profile_df['user_id'] == user_id, f"item_id_{rating}"] = ", ".join(map(str, items_with_rating['item_id'].tolist()))
            else:
                profile_df.loc[profile_df['user_id'] == user_id, f"item_id_{rating}"] = ""
    
    profile_df.to_csv('data/cf_data/user_profile.csv', index=False)
    
    print("데이터 준비 완료!")

if __name__ == "__main__":
    main() 