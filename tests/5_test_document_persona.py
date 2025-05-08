import re
import pandas as pd
import os.path as op
import os

def generate_init_info(s):
    """
    페르소나 텍스트에서 취향, 이유, 높은 평점, 낮은 평점 정보를 추출합니다.
    """
    # 정규 표현식 패턴 정의
    taste_pattern = r'TASTE:(.*?)(?=REASON:|HIGH RATINGS:|LOW RATINGS:|$)'
    reason_pattern = r'REASON:(.*?)(?=TASTE:|HIGH RATINGS:|LOW RATINGS:|$)'
    high_ratings_pattern = r'HIGH RATINGS:(.*?)(?=TASTE:|REASON:|LOW RATINGS:|$)'
    low_ratings_pattern = r'LOW RATINGS:(.*?)(?=TASTE:|REASON:|HIGH RATINGS:|$)'
    
    # 취향 추출
    taste_matches = re.findall(taste_pattern, s, re.DOTALL)
    taste = '|'.join([match.strip() for match in taste_matches if match.strip()])
    
    # 이유 추출
    reason_matches = re.findall(reason_pattern, s, re.DOTALL)
    reason = '|'.join([match.strip() for match in reason_matches if match.strip()])
    
    # 높은 평점 추출
    high_matches = re.findall(high_ratings_pattern, s, re.DOTALL)
    high = '|'.join([match.strip() for match in high_matches if match.strip()])
    
    # 낮은 평점 추출
    low_matches = re.findall(low_ratings_pattern, s, re.DOTALL)
    low = '|'.join([match.strip() for match in low_matches if match.strip()])
    
    return "|".join([taste, reason, high, low])

def document_persona():
    """
    생성된 페르소나를 문서화합니다.
    """
    print("페르소나 문서화 중...")
    
    # 필요한 디렉토리 생성
    os.makedirs('data/simulation', exist_ok=True)
    
    # 페르소나 파일 경로
    base_path = 'data/raw_data/persona_descriptions'
    target_file = op.join(base_path, "persona_0.txt")
    
    # 페르소나 디렉토리 확인
    if not op.exists(base_path):
        print(f"오류: 페르소나 디렉토리({base_path})가 존재하지 않습니다. 먼저 4_test_generate_persona.py를 실행해주세요.")
        return None
    
    # 페르소나 파일 확인
    if not op.exists(target_file):
        print(f"오류: 페르소나 파일({target_file})이 존재하지 않습니다. 먼저 4_test_generate_persona.py를 실행해주세요.")
        return None
    
    # 페르소나 파일 로드
    with open(target_file, 'r', encoding='utf-8') as f:
        persona_text = f.read()
    
    # 페르소나 정보 추출
    persona_info = generate_init_info(persona_text)
    persona_parts = persona_info.split('|')
    
    # 추출된 정보를 DataFrame으로 변환
    if len(persona_parts) >= 4:
        taste, reason, high, low = persona_parts[:4]
        
        persona_df = pd.DataFrame({
            'user_id': [0],
            'taste': [taste],
            'reason': [reason],
            'high': [high],
            'low': [low]
        })
        
        # 결과 저장
        persona_df.to_csv('data/simulation/persona.csv', index=False)
        
        print("페르소나 문서화 완료!")
        return persona_df
    else:
        print("오류: 페르소나 정보가 올바르게 추출되지 않았습니다.")
        return None

def main():
    # 페르소나 문서화
    persona_df = document_persona()
    
    if persona_df is not None:
        print("\n문서화된 페르소나 속성:")
        print("취향:", persona_df.iloc[0]['taste'])
        print("이유:", persona_df.iloc[0]['reason'])
        print("높은 평점:", persona_df.iloc[0]['high'])
        print("낮은 평점:", persona_df.iloc[0]['low'])
    else:
        print("페르소나 문서화에 실패했습니다.")

if __name__ == "__main__":
    main() 