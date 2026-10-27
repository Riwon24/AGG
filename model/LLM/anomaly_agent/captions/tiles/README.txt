aa_caption_tiles: AGG1 에서 직접 표시한 AGG 타일 17개와 AGG0 이미지의 타일 17개를 Claude API 로 캡션 생성하고, 반복되는 표현을 비교한다.

aa_caption_phrases.py 와 같은 방식 (타일마다 관점 3가지 캡션, 라벨 정보는 모델에 알리지 않음)이고, 보내는 단위가 사진 전체가 아니라 224px 타일이다 (448px 로 확대해서 전송).
타일 선택:
  - AGG1(표시) : exp_aggsim/outputs/c08_agg1_manual_labels/agg_tile_labels_filled.csv 에서 has_agg == 1 인 17개 전부.
  - AGG0(대조) : AGG0 이미지의 분석 타일 중 17개. 표시 타일의 안쪽(mask_frac=1)/가장자리 개수와 같게 뽑고 (가장자리는 mask_frac 0.75 이상만), 같은 이미지에서 2개 이상 안 뽑는다 (seed 고정).
실행: ANTHROPIC_API_KEY (필요하면 ANTHROPIC_WORKSPACE_ID) 를 환경변수로 설정한 뒤  python aa_caption_tiles.py [--dry-run]
출력: outputs/aa_caption_tiles/captions.json, phrases.csv, tiles/<그룹>_<번호>.png (보낸 타일 확인용, 로컬 전용), README.txt
주의: 타일 이미지 34장을 외부 API 로 전송한다.

모델: claude-sonnet-5-5
