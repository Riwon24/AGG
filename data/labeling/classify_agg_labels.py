"""하위폴더(6개) x 하위폴더(각 10개) = 60개 폴더의 이미지 파일명에서 AGG 라벨(AGG0/1/2)을 추출해 출력한다.

출력 예)  REF/PH1180-1050: AGG2
"""
import re
import sys
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
# 파일명(확장자 제외) 끝부분의 "- AGG0/1/2" 를 찾는다. 공백 개수, 대소문자 차이는 허용.
LABEL_RE = re.compile(r"-\s*AGG\s*([012])\s*$", re.IGNORECASE)

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent

unlabeled_suspects = []  # 'AGG'는 들어 있지만 규칙에 안 맞는 파일 (오타 점검용)

for group in sorted(p for p in root.iterdir() if p.is_dir()):          # 상위 6개
    for sub in sorted(p for p in group.iterdir() if p.is_dir()):       # 각 10개
        labels = Counter()
        for f in sub.iterdir():
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            m = LABEL_RE.search(f.stem)
            if m:
                labels[f"AGG{m.group(1)}"] += 1
            elif "agg" in f.stem.lower():
                unlabeled_suspects.append(f)

        if not labels:
            result = "(라벨 없음)"
        elif len(labels) == 1:
            result = next(iter(labels))
        else:  # 한 폴더에 서로 다른 라벨이 섞여 있는 경우: 개수 함께 표시
            result = ", ".join(f"{k}({v}장)" for k, v in sorted(labels.items()))
        print(f"{group.name}/{sub.name}: {result}")

if unlabeled_suspects:
    print("\n[확인 필요] 'AGG'가 있지만 형식에 안 맞는 파일:")
    for f in unlabeled_suspects:
        print(f"  {f.relative_to(root)}")
