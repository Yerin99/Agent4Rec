import os
import json

def fix_seeds(seed=101):
    """
    실행 결과 재현을 위해 시드를 고정합니다.
    """
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"시드가 {seed}로 고정되었습니다.")

def create_test_data():
    """
    ver_amazon/data 폴더에서 데이터를 불러와 테스트용 단일 사용자 및 단일 아이템 데이터를 생성합니다.
    """
    # ver_amazon/data 폴더의 JSON 파일 경로
    review_path = "../ver_amazon/data/Movies_and_TV.json"
    meta_path = "../ver_amazon/data/meta_Movies_and_TV.json"
    
    # 출력 디렉토리 생성
    os.makedirs('data/test_data', exist_ok=True)
    
    # 원본 데이터에서 읽어오기
    reviews = []
    with open(review_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # 100개의 리뷰만 로드
                break
            try:
                reviews.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error decoding line {i}: {line[:50]}")
                continue
    
    meta = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 50:  # 50개의 메타데이터만 로드
                break
            try:
                meta.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                print(f"Error decoding line {i}: {line[:50]}")
                continue
    
    # 1명의 사용자 선택 (가장 많은 리뷰를 남긴 사용자)
    reviewer_counts = {}
    for review in reviews:
        reviewer_id = review.get('reviewerID')
        if reviewer_id in reviewer_counts:
            reviewer_counts[reviewer_id] += 1
        else:
            reviewer_counts[reviewer_id] = 1
    
    # 리뷰를 가장 많이 남긴 사용자 선택
    if not reviewer_counts:
        print("리뷰 데이터가 없습니다.")
        return
        
    top_reviewer = max(reviewer_counts.items(), key=lambda x: x[1])[0]
    print(f"선택된 사용자 ID: {top_reviewer}, 리뷰 수: {reviewer_counts[top_reviewer]}")
    
    # 선택된 사용자의 리뷰만 필터링
    selected_reviews = [r for r in reviews if r.get('reviewerID') == top_reviewer]
    
    # 해당 사용자가 리뷰한 아이템의 메타데이터만 필터링
    reviewed_asins = set(r.get('asin') for r in selected_reviews)
    selected_meta = [m for m in meta if m.get('asin') in reviewed_asins]
    
    print(f"리뷰된 아이템 수: {len(reviewed_asins)}")
    print(f"필터링된 메타데이터 수: {len(selected_meta)}")
    
    # 데이터 샘플 출력
    print("\n첫 번째 리뷰 샘플:")
    if selected_reviews:
        print(json.dumps(selected_reviews[0], indent=2))
    
    print("\n첫 번째 메타데이터 샘플:")
    if selected_meta:
        print(json.dumps(selected_meta[0], indent=2))
    
    # 테스트 데이터 저장
    with open('data/test_data/test_reviews.json', 'w', encoding='utf-8') as f:
        json.dump(selected_reviews, f, indent=2)
    
    with open('data/test_data/test_meta.json', 'w', encoding='utf-8') as f:
        json.dump(selected_meta, f, indent=2)
    
    print(f"\n테스트 데이터가 생성되었습니다. 1명의 사용자와 {len(selected_meta)}개의 아이템")

def main():
    fix_seeds()
    create_test_data()
    print("테스트 데이터 생성이 완료되었습니다!")

if __name__ == "__main__":
    main() 