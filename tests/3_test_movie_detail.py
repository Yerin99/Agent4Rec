import numpy as np
import pandas as pd
import os
import json
import re

def clean_description(text):
    """
    제품 설명에서 HTML 태그와 불필요한 문자를 제거합니다.
    """
    if pd.isna(text) or text is None:
        return ""
    
    # HTML 태그 제거
    text = re.sub(r'<.*?>', ' ', str(text))
    
    # 여러 공백을 하나로 변환
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def normalize_asin(asin):
    """
    ASIN을 정규화합니다.
    숫자 형식의 ASIN인 경우 10자리가 되도록 앞에 0을 채웁니다.
    """
    if pd.isna(asin) or asin is None:
        return ""
    
    asin_str = str(asin).strip()
    
    # ASIN이 숫자로만 이루어진 경우 10자리가 되도록 앞에 0을 채움
    if asin_str.isdigit() and len(asin_str) < 10:
        return asin_str.zfill(10)
    
    return asin_str

def process_item_details():
    """
    아이템 세부 정보를 처리합니다.
    메타데이터에서 제목, 설명, 카테고리 등의 정보를 추출하여 저장합니다.
    """
    print("아이템 세부 정보 처리 중...")
    
    # 필요한 디렉토리 생성
    os.makedirs('data/raw_data', exist_ok=True)
    
    # 아이템 데이터 로드 - ASIN을 문자열로 명시적 지정
    items_df = pd.read_csv('data/cf_data/all_items.csv', dtype={'asin': str})
    print(f"로드된 아이템 수: {len(items_df)}")
    print(f"아이템 ASIN 데이터 타입: {items_df['asin'].dtype}")
    
    # ASIN 정규화
    items_df['asin'] = items_df['asin'].apply(normalize_asin)
    
    # 메타데이터 로드 (테스트 데이터에서)
    with open('data/test_data/test_meta.json', 'r', encoding='utf-8') as f:
        meta_data = json.load(f)
    print(f"로드된 메타데이터 수: {len(meta_data)}")
    
    # 메타데이터 ASIN 정규화
    for item in meta_data:
        if 'asin' in item:
            item['asin'] = normalize_asin(item['asin'])
    
    # 메타데이터를 딕셔너리로 변환하여 빠른 검색
    meta_dict = {}
    for item in meta_data:
        asin = item.get('asin', '')
        if asin:
            meta_dict[asin] = item
    
    print(f"메타데이터 ASIN 목록: {list(meta_dict.keys())}")
    
    # 결과를 저장할 리스트
    processed_items = []
    
    # 각 아이템 처리
    for _, item in items_df.iterrows():
        item_id = item['item_id']
        asin = normalize_asin(item['asin'])  # ASIN 정규화
        
        print(f"처리 중인 아이템 ID: {item_id}, ASIN: {asin}")
        
        # 메타데이터에서 아이템 정보 찾기
        item_meta = meta_dict.get(asin, {})
        
        if not item_meta:
            print(f"경고: ASIN '{asin}'에 대한 메타데이터를 찾을 수 없습니다.")
            # 숫자만 포함된 ASIN에 대해 추가 시도: 앞에 0을 채워서 10자리로 만듦
            if asin.isdigit():
                padded_asin = asin.zfill(10)
                if padded_asin != asin and padded_asin in meta_dict:
                    print(f"대체 ASIN '{padded_asin}'에서 메타데이터를 찾았습니다.")
                    item_meta = meta_dict[padded_asin]
                    asin = padded_asin
        
        # 제목 추출
        title = item_meta.get('title', '')
        if not title:
            title = f"Unknown Item {asin}"
        
        # 설명 추출 및 정리
        description = item_meta.get('description', '')
        if isinstance(description, list) and description:
            description = ' '.join(description)
        description = clean_description(description)
        
        # 카테고리 추출
        categories = item_meta.get('category', [])
        if not isinstance(categories, list):
            categories = [categories] if categories else []
        
        # 특성 추출
        # Amazon 데이터 구조에 맞게 조정
        features = []
        for feature_key in ['brand', 'feature']:
            feature_val = item_meta.get(feature_key, None)
            if feature_val:
                if isinstance(feature_val, list):
                    features.extend(feature_val)
                else:
                    features.append(feature_val)
        
        # 가격 추출
        price = item_meta.get('price', '0.0')
        if isinstance(price, str):
            try:
                price = float(price.replace('$', '').replace(',', ''))
            except ValueError:
                price = 0.0
        elif price is None:
            price = 0.0
        
        # 아이템 정보 저장
        processed_item = {
            'item_id': item_id,
            'asin': asin,
            'title': title,
            'description': description,
            'categories': categories,
            'features': features,
            'price': price
        }
        
        processed_items.append(processed_item)
    
    # 결과를 JSON 파일로 저장
    with open('data/raw_data/movie_details.json', 'w', encoding='utf-8') as f:
        json.dump(processed_items, f, ensure_ascii=False, indent=4)
    
    print(f"처리된 아이템 수: {len(processed_items)}")
    if processed_items:
        print(f"샘플 아이템 제목: {processed_items[0]['title']}")
        print(f"샘플 아이템 ASIN: {processed_items[0]['asin']}")
        print(f"샘플 아이템 카테고리: {processed_items[0]['categories']}")
    
    return processed_items

def main():
    # 아이템 세부 정보 처리
    movie_details = process_item_details()
    
    print("아이템 세부 정보 처리 완료!")

if __name__ == "__main__":
    main() 