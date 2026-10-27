"""aa_read_reasons: 에이전트가 타일마다 쓴 판단 이유(reasoner 마지막 판단)와 사용한 도구를 읽기 좋게 출력한다.

실행 (WSL): python aa_read_reasons.py <결과폴더 이름> <타일 번호 ...>
  예) python aa_read_reasons.py agg_tile34_v2 4 9 14 11      (AGG1표시_NN 타일, 번호는 aa_caption_tiles 의 n)
"""
import json
import os
import sys
import textwrap

R = os.path.expanduser("~/AnomalyAgent/results")
name = sys.argv[1] if len(sys.argv) > 1 else "agg_tile34_v2"
nums = [int(v) for v in sys.argv[2:]] or [4, 9, 14, 11]
by = {x["img_path"]: x for x in json.load(open(os.path.join(R, name, "results.json"), encoding="utf-8"))["results"]}
for n in nums:
    x = by[f"AGG1표시_{n:02d}.png"]
    print(f"\n=== AGG1표시_{n:02d} | 판정 {x['pred_result']} | 도구 {[t['tool_name'].replace('image_', '') for t in x['tool_calls']]}")
    for j in x["reasoner_judgments"][-1:]:
        print(f"  [reasoner #{j.get('call_index')}] {j['result']}: {textwrap.shorten(str(j.get('reason')), 700)}")
