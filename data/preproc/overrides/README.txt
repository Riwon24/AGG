overrides/ 사용법 (s02_specimen_mask.py 가 읽는다; 키 = safe_id = image_id 의 '/' 를 '__' 로 바꾼 것)

1) 개별 임계값 보정:  overrides/mask_<safe_id>.json
   {"method": "manual", "s": 40, "v": 70}            # S>=s & V>=v
   {"method": "otsu_v", "open_px": 41, "close_px": 31}  # 방법은 유지하고 모폴로지만 변경
2) 수동 마스크:       overrides/mask_<safe_id>.png    # 흰색(>127)=시편, 원본과 같은 1824x1368
   (png 가 있으면 json 보다 우선한다)

적용 후 s02 -> s03 을 다시 실행한다. s02 의 mask_stats.csv 의 mask_source 열에 override 여부가 남는다.
파일명은 outputs/s03_mask_qc/review_list.csv 의 safe_id 열을 그대로 쓰면 된다.
