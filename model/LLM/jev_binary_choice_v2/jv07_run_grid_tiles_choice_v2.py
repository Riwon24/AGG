"""jv07: jv06_run_grid_tiles_choice.py 와 코드, 설계가 똑같고 설명문(STATE)만 보강한 판. 질문 문구(flat 의 문구와 두 선택지 설명, extent), 타일 목록, 순서 교차, 모델, 호출은 그대로다.

보강 근거 (jv06 의 타일별 점수에서 평가 이미지의 타일을 직접 열어 본 결과, exp_jev/outputs/jv07_tiles_choice_v2/review/ 의 시트):
  - AGG0 이미지에서 p_flat 이 가장 높은 타일 12개는 전부 가장자리 타일이고, 시편 옆의 어둡고 매끈한 배경이 크게 들어 있다.
  - 안쪽 타일만 보면(AGG0 이미지, 점수 상위 12개) 입상 질감은 그대로인데 조명의 밝은 띠와 호, 흠집과 섬유와 이물질, 노랗거나 갈색인 얼룩, 밝기 기울기와 반사가 높은 점수를 받았다.
  - AGG1+2 이미지에서 p_flat 이 가장 낮은 타일 12개는 전부 입상 질감이 고른 정상으로 보여 모델의 오류가 아니다 (타일 라벨이 이미지 라벨의 복사라서 생김). 설명문은 이것 때문에 고치지 않았다.
  그래서 기존 설명문 뒤에 "AGG 는 시편 표면에서만 나타난다"와 위 네 가지 제외 항목을 붙였다. AGG 의 정의 문장은 건드리지 않았다.
주의: 이 보강은 평가에 쓰는 같은 60장의 타일을 보고 만든 것이라 jv07 의 점수는 낙관적일 수 있다.
사용: python jv07_run_grid_tiles_choice_v2.py [--out jsonl] [--limit N]
"""
import sys
from pathlib import Path

import jv06_run_grid_tiles_choice as base

base.STATE = (base.STATE + "\n\n"
              "Only the sample surface can show AGG. Do NOT mark as AGG:\n"
              "- The dark, featureless background outside the sample, or the sample's edge where it meets the dark background "
              "(a crop may show part of the sample and part of the background).\n"
              "- Broad bright or dark bands, arcs, glare or brightness gradients caused by lighting, where the fine granular texture continues unchanged.\n"
              "- Scratches, fibers, dust or other small foreign marks on the surface.\n"
              "- Stains or color patches (for example yellowish or brownish) where the granular texture is still present.")

if __name__ == "__main__":
    if "--out" not in sys.argv:
        sys.argv += ["--out", str(Path.home() / "jevision_work/results/tiles_grid_choice_v2.jsonl")]
    base.main()
