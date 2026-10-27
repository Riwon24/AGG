aa_caption_phrases: AGG0 / AGG2 이미지 각 5장을 Claude API 로 캡션 생성하고, 반복되는 표현을 뽑는다.

AnomalyAgent 의 템플릿 도구가 캡션 3개(관점 3가지)를 쓰므로 이미지마다 서로 다른 관점의 캡션 3개를 받는다.
  관점 1: 표면 전체의 인상  관점 2: 질감과 표면 상태(매끄러움, 거칠기, 반사)  관점 3: 눈에 띄는 영역이나 불규칙한 부분
라벨(AGG0/AGG2)은 모델에게 알려주지 않는다 (캡션이 라벨 이름에 끌려가지 않게).
이미지 선택: 라벨마다 금속 종류가 겹치지 않게 5장 (seed 고정). 긴 변을 1568px 로 줄여서 보낸다.
실행: ANTHROPIC_API_KEY 를 환경변수로 설정한 뒤 (키가 워크스페이스에 연결돼 있지 않으면 ANTHROPIC_WORKSPACE_ID 도)  python aa_caption_phrases.py [--dry-run] [--model claude-sonnet-5-5]
출력: outputs/aa_caption_phrases/captions.json (이미지별 캡션), phrases.csv (표현, 라벨별 등장 캡션 수), README.txt
표현 추출: 소문자, 2~4단어 연속 구(n-gram), 같은 구가 서로 다른 캡션에 몇 번 나왔는지(캡션 수 기준)로 센다.
주의: 이미지 10장을 외부 API 로 전송한다 (원본 사진은 저장소에 올리지 않는 것과 별개로, 이 실행에서만 전송됨).

모델: claude-sonnet-5-5
