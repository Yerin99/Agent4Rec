import numpy as np
import pandas as pd
import os
import random
import torch
import pickle
import json
import time
from dotenv import load_dotenv

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

def get_completion(api_key, model, prompt, temperature=0, max_tokens=1000):
    """
    OpenAI API를 사용하여 프롬프트에 대한 응답을 생성합니다.
    OpenAI 라이브러리 버전에 따라 호환되도록 구현합니다.
    """
    import openai
    
    # OpenAI API 키 설정
    openai.api_key = api_key
    
    # OpenAI 버전 확인
    if hasattr(openai, 'OpenAI'):
        # OpenAI >= 1.0.0
        print("OpenAI API v1.0 이상 사용")
        client = openai.OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message.content
        print(f"입력 토큰: {response.usage.prompt_tokens}, 출력 토큰: {response.usage.completion_tokens}, 총 토큰: {response.usage.total_tokens}")
        
    else:
        # OpenAI < 1.0.0
        print("OpenAI API v0.x 사용")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        result = response.choices[0].message["content"]
        print(f"입력 토큰: {response.usage.prompt_tokens}, 출력 토큰: {response.usage.completion_tokens}, 총 토큰: {response.usage.total_tokens}")
    
    return result

def generate_persona():
    """
    사용자 페르소나를 생성합니다.
    """
    print("사용자 페르소나 생성 중...")
    
    # 필요한 디렉토리 생성
    persona_dir = 'data/raw_data/persona_descriptions'
    os.makedirs(persona_dir, exist_ok=True)
    
    # OpenAI API 키 로드
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OpenAI API 키가 설정되지 않았습니다. .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        return False
    
    model = "gpt-3.5-turbo"
    
    # 사용자 프로필 데이터 로드
    try:
        user_profile = pd.read_csv('data/cf_data/user_profile.csv')
        print(f"사용자 프로필 로드 완료: {len(user_profile)}행")
    except FileNotFoundError:
        print("사용자 프로필 파일을 찾을 수 없습니다. 먼저 1_test_cf_data.py를 실행해주세요.")
        return False
    
    # 평점 데이터 로드
    try:
        ratings = pd.read_csv('data/cf_data/all_ratings.csv', dtype={'asin': str})
        print(f"평점 데이터 로드 완료: {len(ratings)}행")
        # ASIN 정규화
        ratings['asin'] = ratings['asin'].apply(normalize_asin)
    except FileNotFoundError:
        print("평점 데이터 파일을 찾을 수 없습니다. 먼저 1_test_cf_data.py를 실행해주세요.")
        return False
    
    # 아이템 정보 로드
    try:
        items_df = pd.read_csv('data/cf_data/all_items.csv', dtype={'asin': str})
        print(f"아이템 정보 로드 완료: {len(items_df)}행")
        # ASIN 정규화
        items_df['asin'] = items_df['asin'].apply(normalize_asin)
    except FileNotFoundError:
        print("아이템 정보 파일을 찾을 수 없습니다. 먼저 1_test_cf_data.py를 실행해주세요.")
        return False
    
    # 아이템 상세 정보 로드
    try:
        with open('data/raw_data/movie_details.json', 'r', encoding='utf-8') as f:
            movie_details = json.load(f)
        print(f"영화 상세 정보 로드 완료: {len(movie_details)}개 항목")
        
        # ASIN 정규화
        for item in movie_details:
            if 'asin' in item:
                item['asin'] = normalize_asin(item['asin'])
    except FileNotFoundError:
        print("영화 상세 정보 파일을 찾을 수 없습니다. 먼저 3_test_movie_detail.py를 실행해주세요.")
        return False
    
    # 아이템 정보 딕셔너리 생성 (item_id 기준)
    movie_dict_by_id = {}
    for item in movie_details:
        item_id = item.get('item_id')
        if item_id is not None:
            movie_dict_by_id[item_id] = item
    
    # 아이템 정보 딕셔너리 생성 (asin 기준)
    movie_dict_by_asin = {}
    for item in movie_details:
        asin = item.get('asin')
        if asin:
            movie_dict_by_asin[asin] = item
    
    print(f"영화 정보 딕셔너리 생성 완료: ID 기준 {len(movie_dict_by_id)}개, ASIN 기준 {len(movie_dict_by_asin)}개 항목")
    print(f"영화 정보 딕셔너리 키(ID): {list(movie_dict_by_id.keys())}")
    print(f"영화 정보 딕셔너리 키(ASIN): {list(movie_dict_by_asin.keys())}")
    
    # 프롬프트 생성
    user_id = 0  # 단일 사용자
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('overall', ascending=False)
    
    high_rated_items = []
    low_rated_items = []
    
    # 높은 평점과 낮은 평점 아이템 수집
    for _, row in user_ratings.iterrows():
        item_id = row['item_id']
        asin = normalize_asin(row['asin'])
        rating = row['overall']
        
        print(f"처리 중인 아이템 - ID: {item_id}, ASIN: {asin}, 평점: {rating}")
        
        # 아이템 정보 찾기 (ID 기준)
        item_info = None
        if item_id in movie_dict_by_id:
            item_info = movie_dict_by_id[item_id]
        # 아이템 정보 찾기 (ASIN 기준)
        elif asin in movie_dict_by_asin:
            item_info = movie_dict_by_asin[asin]
        
        if item_info:
            item_title = item_info.get('title', '')
            item_categories = item_info.get('categories', [])
            
            item_entry = f"{item_title} (카테고리: {', '.join(item_categories) if item_categories else '정보 없음'}, 평점: {rating})"
            
            if rating >= 4.0:
                high_rated_items.append(item_entry)
            elif rating <= 2.0:
                low_rated_items.append(item_entry)
        else:
            # 영화 세부 정보에서 찾지 못한 경우 아이템 기본 정보에서 찾기
            item_info = items_df[items_df['asin'] == asin]
            if not item_info.empty:
                item_title = item_info['title'].iloc[0]
                categories = item_info.get('category', [])
                if not isinstance(categories, list):
                    categories = []
                
                item_entry = f"{item_title} (카테고리: {', '.join(categories) if categories else '정보 없음'}, 평점: {rating})"
                
                if rating >= 4.0:
                    high_rated_items.append(item_entry)
                elif rating <= 2.0:
                    low_rated_items.append(item_entry)
            else:
                print(f"경고: 아이템 ID {item_id}, ASIN {asin}에 대한 정보를 찾을 수 없습니다.")
    
    print(f"높은 평점 아이템: {len(high_rated_items)}개, 낮은 평점 아이템: {len(low_rated_items)}개")
    
    # 프롬프트 템플릿
    prompt_template = f"""
I want you to act as an agent. You will act as a product taste analyst.
Given a user's rating history from Amazon Movies & TV products:

user gives a rating of 1-2 (low ratings) for following products:
{chr(10).join(low_rated_items[:10]) if low_rated_items else "None"}

user gives a rating of 4-5 (high ratings) for following products:
{chr(10).join(high_rated_items[:10]) if high_rated_items else "None"}

My first request is "I need help creating product taste for a user given the rating history. (in no particular order)"  Generate as many TASTE-REASON pairs as possible, taste should focus on the products' categories and features.
Strictly follow the output format below:

TASTE: <-descriptive taste->
REASON: <-brief reason->

TASTE: <-descriptive taste->
REASON: <-brief reason->
.....

Secondly, analyze what kinds of products the user tends to give high ratings to, and what kinds of products they tend to give low ratings to.
Strictly follow the output format below:
HIGH RATINGS: <-conclusion of products with high ratings->
LOW RATINGS: <-conclusion of products with low ratings->

Answer should be based strictly on the provided data and observed patterns, not on assumptions. Do not include product names in your analysis.
"""
    
    # 페르소나 생성
    print("OpenAI API를 통해 페르소나 생성 중...")
    try:
        response = get_completion(api_key, model, prompt_template)
        
        if response:
            # 페르소나 저장
            persona_file = os.path.join(persona_dir, f"persona_{user_id}.txt")
            with open(persona_file, 'w', encoding='utf-8') as f:
                f.write(response)
            
            print(f"페르소나가 생성되어 {persona_file}에 저장되었습니다.")
            return True
        else:
            print("페르소나 생성에 실패했습니다: API 응답이 없습니다.")
            return False
    except Exception as e:
        print(f"OpenAI API 호출 중 오류 발생: {e}")
        return False

def main():
    seed = 101
    fix_seeds(seed)
    
    # 페르소나 생성
    success = generate_persona()
    
    if success:
        print("사용자 페르소나 생성 완료!")
    else:
        print("사용자 페르소나 생성 실패.")

if __name__ == "__main__":
    main() 