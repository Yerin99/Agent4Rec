import re
import pandas as pd
import os.path as op
import os

def generate_init_info(s):
    """
    페르소나 텍스트에서 취향(TASTE), 이유(REASON) 정보를 추출합니다.
    """
    # 정규 표현식 패턴 정의
    taste_pattern = r'TASTE:(.*?)(?=REASON:|$)'
    reason_pattern = r'REASON:(.*?)(?=TASTE:|$)'
    
    # 취향 추출
    taste_matches = re.findall(taste_pattern, s, re.DOTALL)
    taste = '|'.join([match.strip() for match in taste_matches if match.strip()])
    
    # 이유 추출
    reason_matches = re.findall(reason_pattern, s, re.DOTALL)
    reason = '|'.join([match.strip() for match in reason_matches if match.strip()])
    
    return taste, reason

def document_persona():
    """
    생성된 페르소나를 문서화합니다.
    """
    print("페르소나 문서화 중...")
    
    # 필요한 디렉토리 생성
    os.makedirs('data/simulation', exist_ok=True)
    
    # 페르소나 파일 디렉토리 확인
    base_path = 'data/raw_data/persona_descriptions'
    if not op.exists(base_path):
        print(f"오류: 페르소나 디렉토리({base_path})가 존재하지 않습니다.")
        return None
    
    # 페르소나 파일 검색
    persona_files = [f for f in os.listdir(base_path) if f.startswith('persona_') and f.endswith('.txt')]
    
    if not persona_files:
        print(f"오류: 페르소나 파일이 없습니다. 먼저 4_generate_persona.py를 실행해주세요.")
        return None
    
    print(f"처리할 페르소나 파일: {len(persona_files)}개")
    
    # 결과를 저장할 데이터프레임 준비
    persona_data = []
    
    for persona_file in persona_files:
        # 사용자 ID 추출
        try:
            user_id = int(persona_file.split('_')[1].split('.')[0])
        except ValueError:
            # 파일명에서 사용자 ID를 추출할 수 없는 경우 스킵
            print(f"경고: {persona_file}에서 사용자 ID를 추출할 수 없습니다.")
            continue
        
        # 페르소나 파일 로드
        file_path = op.join(base_path, persona_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            persona_text = f.read()
        
        # 페르소나 정보 추출
        taste, reason = generate_init_info(persona_text)
        
        # 데이터프레임에 추가
        persona_data.append({
            'user_id': user_id,
            'taste': taste,
            'reason': reason
        })
    
    # 페르소나 데이터프레임 생성
    persona_df = pd.DataFrame(persona_data)
    
    # 결과 저장
    persona_df.to_csv('data/simulation/persona.csv', index=False)
    
    print("페르소나 문서화 완료!")
    print(f"총 {len(persona_df)}명의 사용자 페르소나가 문서화되었습니다.")
    
    return persona_df

def main():
    # 페르소나 문서화
    persona_df = document_persona()
    
    if persona_df is not None:
        print("\n문서화된 페르소나 속성 (첫 번째 사용자):")
        if not persona_df.empty:
            first_user = persona_df.iloc[0]
            print(f"사용자 ID: {first_user['user_id']}")
            print("취향 예시:")
            if '|' in first_user['taste']:
                tastes = first_user['taste'].split('|')
                for i, taste in enumerate(tastes[:3]):  # 최대 3개만 출력
                    print(f"  {i+1}. {taste}")
            else:
                print(f"  {first_user['taste']}")
            
            print(f"이유 예시:")
            if '|' in first_user['reason']:
                reasons = first_user['reason'].split('|')
                for i, reason in enumerate(reasons[:3]):  # 최대 3개만 출력
                    print(f"  {i+1}. {reason}")
            else:
                print(f"  {first_user['reason']}")
    else:
        print("페르소나 문서화에 실패했습니다.")

if __name__ == "__main__":
    main() 