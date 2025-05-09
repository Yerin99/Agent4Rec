import numpy as np
import pandas as pd
import os
import random
import torch
import pickle
import json
import time
from dotenv import load_dotenv
import openai
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor

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

def get_completion(api_key, model, prompt, sys_prompt="You are a helpful assistant.", temperature=0.2, max_tokens=1000):
    """
    OpenAI API를 사용하여 프롬프트에 대한 응답을 생성합니다.
    OpenAI 라이브러리 버전에 따라 호환되도록 구현합니다.
    """
    # OpenAI API 키 설정
    openai.api_key = api_key
    
    # 메시지 구성
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt}
    ]
    
    # API 호출 및 재시도 로직
    max_retries = 3
    retry_delay = 5
    
    for attempt in range(max_retries):
        try:
            # OpenAI 버전 확인 및 API 호출
            if hasattr(openai, 'OpenAI'):
                # OpenAI >= 1.0.0
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                result = response.choices[0].message.content
                tokens = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            else:
                # OpenAI < 1.0.0
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                result = response.choices[0].message["content"]
                tokens = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            
            print(f"사용된 토큰: {tokens['total_tokens']/1000:.2f}k")
            print(f"프롬프트 토큰: {tokens['prompt_tokens']/1000:.2f}k")
            print(f"응답 토큰: {tokens['completion_tokens']/1000:.2f}k")
            
            return result
            
        except Exception as e:
            print(f"API 호출 오류 (시도 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"{retry_delay}초 후 재시도합니다...")
                time.sleep(retry_delay)
                retry_delay *= 2
    
    # 모든 재시도 실패 시
    raise Exception("API 호출에 실패했습니다. 나중에 다시 시도해주세요.")

async def generate_persona_for_user(user_id, api_key, model, sys_prompt, prompt_template, high_rated_items, executor, loop):
    """
    특정 사용자에 대한 페르소나를 생성합니다. 비동기 함수입니다.
    """
    print(f"사용자 {user_id}의 페르소나 생성 중...")
    
    # 프롬프트 작성
    prompt = prompt_template.format(
        INPUT1='\n'.join(high_rated_items[:5]) if high_rated_items else " ",
        INPUT2='\n'.join(high_rated_items[5:10]) if len(high_rated_items) > 5 else " "
    )
    
    start_time = time.time()
    try:
        # 비동기로 API 호출 실행
        response = await loop.run_in_executor(
            executor, 
            get_completion, 
            api_key, 
            model, 
            prompt, 
            sys_prompt, 
            0.2
        )
        
        end_time = time.time()
        print(f"사용자 {user_id} 페르소나 생성 완료 (소요 시간: {end_time - start_time:.2f}초)")
        
        # 페르소나 저장
        persona_dir = 'data/raw_data/persona_descriptions'
        os.makedirs(persona_dir, exist_ok=True)
        
        persona_file = os.path.join(persona_dir, f"persona_{user_id}.txt")
        with open(persona_file, 'w', encoding='utf-8') as f:
            f.write(response)
        
        return True
    except Exception as e:
        end_time = time.time()
        print(f"사용자 {user_id} 페르소나 생성 실패 (소요 시간: {end_time - start_time:.2f}초): {e}")
        return False

async def generate_personas():
    """
    모든 사용자에 대한 페르소나를 생성합니다. ThreadPoolExecutor를 사용하여 병렬 처리합니다.
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
        print("사용자 프로필 파일을 찾을 수 없습니다. 먼저 1_get_cf_data.py를 실행해주세요.")
        return False
    
    # 아이템 상세 정보 로드
    try:
        with open('data/raw_data/movie_details.json', 'r', encoding='utf-8') as f:
            movie_details = json.load(f)
        print(f"상품 상세 정보 로드 완료: {len(movie_details)}개 항목")
        
        # ASIN 정규화
        for item in movie_details:
            if 'asin' in item:
                item['asin'] = normalize_asin(item['asin'])
    except FileNotFoundError:
        print("상품 상세 정보 파일을 찾을 수 없습니다. 먼저 3_get_movie_detail.py를 실행해주세요.")
        return False
    
    # 평점 데이터 로드
    try:
        ratings = pd.read_csv('data/cf_data/all_ratings.csv', dtype={'asin': str})
        print(f"평점 데이터 로드 완료: {len(ratings)}행")
        # ASIN 정규화
        ratings['asin'] = ratings['asin'].apply(normalize_asin)
    except FileNotFoundError:
        print("평점 데이터 파일을 찾을 수 없습니다. 먼저 1_get_cf_data.py를 실행해주세요.")
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
    
    print(f"상품 정보 딕셔너리 생성 완료: ID 기준 {len(movie_dict_by_id)}개, ASIN 기준 {len(movie_dict_by_asin)}개 항목")
    
    # 최대 10명의 사용자만 처리
    max_users = min(10, len(user_profile))
    
    # prompt_information_house 템플릿 (원본 코드와 동일)
    sys_prompt = "I want you to act as an agent. You will act as a product taste analyst."
    
    prompt_information_house = """
Given a user's rating history:

user gives high ratings for following products: {INPUT1} {INPUT2}

My first request is "I need help creating product taste for a user given the rating history. (in no particular order)"  
Generate two specific and most inclusive TASTE-REASON pairs as possible, taste should focus on the products' categories and don't use obcure words like "have diverse taste".
Don't conclude the taste using any time-related word like 90's or classic.
Strictly follow the output format below:

TASTE: <-descriptive taste->
REASON: <-brief reason->

TASTE: <-descriptive taste->
REASON: <-brief reason->
"""
    
    # 비동기 처리를 위한 설정
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=10)  # 최대 사용자 수에 맞춰 설정
    tasks = []
    
    # 각 사용자에 대한 작업 생성
    for user_idx in range(max_users):
        user_id = user_profile.iloc[user_idx]['user_id']
        
        user_ratings = ratings[ratings['user_id'] == user_id].sort_values('overall', ascending=False)
        
        high_rated_items = []
        
        # 높은 평점 아이템만 수집 (4-5점)
        for _, row in user_ratings.iterrows():
            item_id = row['item_id']
            asin = normalize_asin(row['asin'])
            rating = row['overall']
            
            if rating >= 4.0:
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
                    item_features = item_info.get('features', [])
                    
                    category_text = ', '.join(item_categories) if item_categories else '정보 없음'
                    feature_text = ', '.join(item_features[:3]) if item_features else '정보 없음'
                    
                    item_entry = f"{item_title} (카테고리: {category_text}, 특징: {feature_text}, 평점: {rating})"
                    high_rated_items.append(item_entry)
        
        print(f"사용자 {user_id}의 높은 평점 아이템: {len(high_rated_items)}개")
        
        # 페르소나 생성 작업 추가
        tasks.append(
            generate_persona_for_user(
                user_id,
                api_key,
                model,
                sys_prompt,
                prompt_information_house,
                high_rated_items,
                executor,
                loop
            )
        )
    
    # 작업 실행
    t_start = time.time()
    results = await asyncio.gather(*tasks)
    t_end = time.time()
    
    success_count = sum(results)
    print(f"\n총 {len(tasks)}명의 사용자 중 {success_count}명의 페르소나가 성공적으로 생성되었습니다.")
    print(f"총 소요 시간: {t_end - t_start:.2f}초")
    
    return success_count > 0

def main():
    seed = 101
    fix_seeds(seed)
    
    # 비동기 함수 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    success = loop.run_until_complete(generate_personas())
    
    if success:
        print("사용자 페르소나 생성 완료!")
    else:
        print("사용자 페르소나 생성 실패.")

if __name__ == "__main__":
    main()