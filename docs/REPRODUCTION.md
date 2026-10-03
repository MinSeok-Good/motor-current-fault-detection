# 재현성 실행 안내

## 입력 파일

같은 폴더에 `ai_ready_time_features.csv`(3,000행)와 `ai_ready_time_features_clean.csv`(2,993행)를 둡니다. 원본 및 특징 CSV는 외부 데이터이므로 저장소에 포함하지 않았습니다. ZIP 대조에는 AI Hub Validation의 `current.zip`이 추가로 필요합니다.

## Windows 예시

Anaconda Prompt에서 `motor-check`를 활성화한 뒤 저장소 루트로 이동합니다.

```bat
conda activate motor-check
pip install -r requirements.txt
python src/reproduce.py --data-dir "C:\Users\USER\125.기계시설물_고장_예지_센서"
python src/verify_raw_zip.py --zip "C:\Users\USER\125.기계시설물_고장_예지_센서\01.데이터\2._Validation\current.zip" --data-dir "C:\Users\USER\125.기계시설물_고장_예지_센서"
```

모델만 실행하려면 `--stage models`, 회귀만 실행하려면 `--stage statistics`를 추가합니다. ZIP 전체 압축 해제는 필요 없습니다. 파일명이 중복되거나 파일이 없으면 해당 행을 오류로 기록합니다. 모든 대상 파일의 결과를 CSV로 저장하며, 불일치나 오류가 있으면 종료 코드 1을 반환합니다.

## 노트북

`02_verify_from_clean_csv.ipynb`는 고정된 기존 날짜 분할, 모델/특징 비교, LODO, SHAP 중요도를 계산합니다. `03_verify_statistics_and_environment.ipynb`는 필터 전후 차분, 날짜 통제 회귀와 환경 정보를 기록합니다. 첫 셀의 `DATA`를 실제 CSV 폴더로 지정하거나, 노트북 실행 전에 `MOTOR_FEATURE_DIR` 환경변수를 지정합니다. `reproduce.py`는 이 노트북의 코드 셀을 그대로 실행하므로 별도의 분석 구현을 유지하지 않습니다.

## 저장된 검증 기록

`results/reproduction/`의 모델 결과 CSV는 사용자가 실행한 02 노트북의 출력에서 추출했습니다. 회귀 결과 CSV는 보조 실행 환경에서 계산한 것으로, 사용자의 03 출력에서도 계수 6개가 허용 오차 내에 일치했습니다. `verified_environment.json`은 사용자의 성공한 03 실행에서 기록한 주요 패키지 버전입니다. 전체 의존성 lock은 아니며, 다른 환경에서 같은 수치를 보장하지 않습니다.

`raw_zip_summary.json`은 사용자가 공유한 전체 ZIP 검증 요약입니다. 개별 3,000행 보고서는 업로드받지 않았으므로 저장소에 포함하지 않았습니다. 실행 시 `raw_zip_verification.csv`가 생성됩니다.

기존 RandomForest와 SHAP의 세부 수치 차이는 README에 기록했습니다. 02의 저장된 출력은 경로 설정을 저장소용으로 바꾸기 전 검증 실행의 기록입니다. 수치 계산과 날짜 분할은 유지했습니다. 로컬 환경의 Python, 패키지 버전은 재실행 결과와 함께 확인해야 합니다.

## 기존 src와의 관계

`preprocessing.py`는 기존 특징 열 외에 날짜와 품질 관련 열을 추가합니다. 기존 25열 CSV와 스키마가 같지 않으며, 이미 검증한 특징값의 전체 대조에는 `verify_raw_zip.py`를 사용합니다. `modeling.py`는 clean CSV의 파일명 날짜를 사용하도록 수정했습니다. 전체 비교를 실행하는 권장 경로는 `reproduce.py`입니다. 기존 `shap_analysis.py`는 새로 학습한 모델의 그림을 생성하며, 최초 분석 그림의 수치 재현을 보장하지 않습니다.
