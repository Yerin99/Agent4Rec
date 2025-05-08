import os
import sys
import time
import subprocess
import traceback

def run_script(script_path):
    """
    Python 스크립트를 실행하고 결과와 실행 시간을 반환합니다.
    
    Args:
        script_path (str): 실행할 스크립트의 경로
        
    Returns:
        tuple: (성공 여부, 실행 시간, 출력 텍스트)
    """
    print(f"\n{'='*50}")
    print(f"실행 중: {script_path}")
    print(f"{'='*50}")
    
    start_time = time.time()
    
    try:
        # Python 스크립트 실행
        result = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 출력 결과 표시
        print(f"\n[출력]:")
        print(result.stdout)
        
        # 오류가 있는 경우
        if result.returncode != 0:
            print(f"\n[오류]:")
            print(result.stderr)
            print(f"\n스크립트 실행 실패 ({execution_time:.2f}초)")
            return False, execution_time, result.stdout + "\n" + result.stderr
        
        print(f"\n스크립트 실행 완료 ({execution_time:.2f}초)")
        return True, execution_time, result.stdout
    
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"\n[예외 발생]:")
        traceback.print_exc()
        print(f"\n스크립트 실행 실패 ({execution_time:.2f}초)")
        
        return False, execution_time, str(e)

def main():
    """
    모든 테스트 스크립트를 순차적으로 실행합니다.
    """
    print("Amazon 데이터셋 테스트 스크립트 실행 시작")
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 현재 디렉토리를 tests 폴더로 변경
    current_dir = os.getcwd()
    if os.path.basename(current_dir) != "tests":
        tests_dir = os.path.join(current_dir, "tests")
        if os.path.exists(tests_dir):
            os.chdir(tests_dir)
            print(f"작업 디렉토리를 {tests_dir}로 변경했습니다.")
        else:
            print(f"오류: tests 디렉토리를 찾을 수 없습니다: {tests_dir}")
            return
    
    # 실행할 스크립트 목록
    scripts = [
        "0_test_amazon_data.py",
        "1_test_cf_data.py",
        "2_test_user_statistic.py",
        "3_test_movie_detail.py",
        "4_test_generate_persona.py",
        "5_test_document_persona.py"
    ]
    
    # 각 스크립트 실행 결과 저장
    results = {}
    total_start_time = time.time()
    
    # 스크립트 순차 실행
    for script in scripts:
        if not os.path.exists(script):
            print(f"\n오류: 스크립트 파일을 찾을 수 없습니다: {script}")
            print("이후 스크립트 실행을 중단합니다.")
            break
        
        success, execution_time, output = run_script(script)
        results[script] = {
            "success": success,
            "execution_time": execution_time,
            "output": output
        }
        
        # 실행 실패 시 중단
        if not success:
            print(f"\n오류: {script} 실행 중 오류가 발생했습니다.")
            print("이후 스크립트 실행을 중단합니다.")
            break
    
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time
    
    # 실행 결과 요약
    print("\n" + "="*50)
    print("테스트 실행 결과 요약")
    print("="*50)
    
    for script, result in results.items():
        status = "성공" if result["success"] else "실패"
        print(f"{script}: {status} ({result['execution_time']:.2f}초)")
    
    print(f"\n총 실행 시간: {total_execution_time:.2f}초")
    print(f"종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 원래 디렉토리로 복귀
    if os.getcwd() != current_dir:
        os.chdir(current_dir)

if __name__ == "__main__":
    main() 