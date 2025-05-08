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

def load_test_data():
    """
    테스트용 데이터를 로드합니다.
    """
    with open('data/test_data/test_reviews.json', 'r', encoding='utf-8') as f:
        reviews = json.load(f)
    
    with open('data/test_data/test_meta.json', 'r', encoding='utf-8') as f:
        items = json.load(f)
    
    print(f"리뷰 데이터 로드 완료: {len(reviews)}개")
    print(f"아이템 메타데이터 로드 완료: {len(items)}개")
    
    return reviews, items

def create_cf_data(reviews, items):
    """
    협업 필터링을 위한 데이터를 준비합니다.
    """
    # 리뷰 데이터를 데이터프레임으로 변환
    reviews_df = pd.DataFrame(reviews)
    
    # ASIN을 문자열로 명시적 변환
    reviews_df['asin'] = reviews_df['asin'].astype(str)
    
    # 아이템 데이터를 데이터프레임으로 변환
    items_df = pd.DataFrame(items)
    
    # ASIN을 문자열로 명시적 변환
    if 'asin' in items_df.columns:
        items_df['asin'] = items_df['asin'].astype(str)
    
    # 사용자 ID와 아이템 ID 매핑 (정수로 변환)
    user_id = reviews_df['reviewerID'].unique()[0]
    user_id_map = {user_id: 0}  # 단일 사용자이므로 ID를 0으로 매핑
    
    item_asins = reviews_df['asin'].unique()
    item_id_map = {asin: idx for idx, asin in enumerate(item_asins)}
    
    print(f"사용자 ID 매핑: {user_id_map}")
    print(f"아이템 ID 매핑: {item_id_map}")
    
    # 매핑 정보 저장
    os.makedirs('data/cf_data', exist_ok=True)
    with open('data/cf_data/user_id_map.pkl', 'wb') as f:
        pickle.dump(user_id_map, f)
    
    with open('data/cf_data/item_id_map.pkl', 'wb') as f:
        pickle.dump(item_id_map, f)
    
    # 매핑된 ID로 리뷰 데이터 변환
    reviews_df['user_id'] = reviews_df['reviewerID'].map(user_id_map)
    reviews_df['item_id'] = reviews_df['asin'].map(item_id_map)
    
    # 긍정적인 평가만 필터링 (평점 > 3)
    positive_reviews = reviews_df[reviews_df['overall'] > 3]
    
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
    train_user_dict = {}
    valid_user_dict = {}
    test_user_dict = {}
    
    for user, item in train_interactions:
        if user not in train_user_dict:
            train_user_dict[user] = []
        train_user_dict[user].append(item)
    
    for user, item in valid_interactions:
        if user not in valid_user_dict:
            valid_user_dict[user] = []
        valid_user_dict[user].append(item)
    
    for user, item in test_interactions:
        if user not in test_user_dict:
            test_user_dict[user] = []
        test_user_dict[user].append(item)
    
    # 사용자-아이템 상호작용을 텍스트 파일로 저장
    def save_user_dict_to_txt(user_dict, filepath):
        with open(filepath, 'w') as f:
            for u, items in user_dict.items():
                f.write(str(int(u)))
                for i in items:
                    f.write(' ' + str(int(i)))
                f.write('\n')
    
    save_user_dict_to_txt(train_user_dict, 'data/cf_data/train.txt')
    save_user_dict_to_txt(valid_user_dict, 'data/cf_data/valid.txt')
    save_user_dict_to_txt(test_user_dict, 'data/cf_data/test.txt')
    
    # 평점별 아이템 분류
    rating_groups = reviews_df.groupby('overall')
    
    rating_item_lists = {}
    for rating, group in rating_groups:
        rating_item_lists[rating] = group['asin'].tolist()
    
    # 평점별 아이템 목록 저장
    with open('data/cf_data/rating_item_lists.pkl', 'wb') as f:
        pickle.dump(rating_item_lists, f)
    
    # 모든 평점 데이터 저장 (페르소나 생성에 사용)
    # ASIN을 문자열로 유지하기 위해 dtypes 지정
    reviews_df[['user_id', 'item_id', 'overall', 'reviewText', 'summary', 'asin']].to_csv(
        'data/cf_data/all_ratings.csv', index=False)
    
    # 아이템 정보 저장
    items_df = items_df[['asin', 'title', 'main_cat', 'category']].copy()
    # ASIN 문자열 타입 유지
    items_df['asin'] = items_df['asin'].astype(str)
    items_df['item_id'] = items_df['asin'].map(item_id_map)
    items_df = items_df.dropna(subset=['item_id'])
    
    # 아이템 정보 디버깅 출력
    print("\n아이템 정보:")
    print(items_df)
    print(f"아이템 ASIN 데이터 타입: {items_df['asin'].dtype}")
    
    # 명시적으로 dtypes 지정하여 저장
    items_df.to_csv('data/cf_data/all_items.csv', index=False)
    
    print("협업 필터링 데이터 준비 완료!")
    print(f"사용자 수: 1")
    print(f"아이템 수: {len(item_id_map)}")
    print(f"훈련 상호작용 수: {len(train_interactions)}")
    print(f"검증 상호작용 수: {len(valid_interactions)}")
    print(f"테스트 상호작용 수: {len(test_interactions)}")
    
    return reviews_df, items_df, rating_item_lists

def main():
    seed = 101
    fix_seeds(seed)
    
    # 필요한 디렉토리 생성
    os.makedirs('data/test_data', exist_ok=True)
    os.makedirs('data/cf_data', exist_ok=True)
    
    print("테스트 데이터 로드 중...")
    reviews, items = load_test_data()
    
    print("협업 필터링 데이터 준비 중...")
    reviews_df, items_df, rating_item_lists = create_cf_data(reviews, items)
    
    print("초기 프로필 데이터 생성 중...")
    # 상위 25개 항목으로 초기 프로필 생성
    n_for_init = min(25, len(reviews))
    
    # 사용자의 평점별 아이템 목록 생성
    user_profile = {}
    for rating in range(1, 6):
        items_with_rating = reviews_df[reviews_df['overall'] == rating]
        if not items_with_rating.empty:
            # 아이템 제목 가져오기
            items_with_details = pd.merge(
                items_with_rating, 
                items_df, 
                on='asin', 
                how='inner'
            )
            
            print(f"\n평점 {rating}에 해당하는 아이템:")
            print(items_with_details)
            
            if not items_with_details.empty:
                titles = items_with_details['title_y' if 'title_y' in items_with_details.columns else 'title'].tolist()
                user_profile[rating] = "; ".join(titles[:n_for_init])
            else:
                user_profile[rating] = "None"
        else:
            user_profile[rating] = "None"
    
    # 초기 프로필 저장
    profile_df = pd.DataFrame([user_profile])
    profile_df['user_id'] = 0  # 단일 사용자
    
    # 아이템 ID도 함께 저장
    rating_items = {}
    for rating in range(1, 6):
        items_with_rating = reviews_df[reviews_df['overall'] == rating]
        if not items_with_rating.empty:
            rating_items[f"item_id_{rating}"] = ", ".join(map(str, items_with_rating['item_id'].tolist()))
        else:
            rating_items[f"item_id_{rating}"] = ""
    
    for key, value in rating_items.items():
        profile_df[key] = value
    
    print("\n사용자 프로필:")
    print(profile_df)
    
    profile_df.to_csv('data/cf_data/user_profile.csv', index=False)
    
    print("데이터 준비 완료!")

if __name__ == "__main__":
    main() 