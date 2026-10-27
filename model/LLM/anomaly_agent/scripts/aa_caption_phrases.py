"""aa_caption_phrases: AGG0 / AGG2 이미지 각 5장을 Claude API 로 캡션 생성하고, 반복되는 표현을 뽑는다.

AnomalyAgent 의 템플릿 도구가 캡션 3개(관점 3가지)를 쓰므로 이미지마다 서로 다른 관점의 캡션 3개를 받는다.
  관점 1: 표면 전체의 인상  관점 2: 질감과 표면 상태(매끄러움, 거칠기, 반사)  관점 3: 눈에 띄는 영역이나 불규칙한 부분
라벨(AGG0/AGG2)은 모델에게 알려주지 않는다 (캡션이 라벨 이름에 끌려가지 않게).
이미지 선택: 라벨마다 금속 종류가 겹치지 않게 5장 (seed 고정). 긴 변을 1568px 로 줄여서 보낸다.
실행: ANTHROPIC_API_KEY 를 환경변수로 설정한 뒤 (키가 워크스페이스에 연결돼 있지 않으면 ANTHROPIC_WORKSPACE_ID 도)  python aa_caption_phrases.py [--dry-run] [--model claude-sonnet-5-5]
출력: outputs/aa_caption_phrases/captions.json (이미지별 캡션), phrases.csv (표현, 라벨별 등장 캡션 수), README.txt
표현 추출: 소문자, 2~4단어 연속 구(n-gram), 같은 구가 서로 다른 캡션에 몇 번 나왔는지(캡션 수 기준)로 센다.
주의: 이미지 10장을 외부 API 로 전송한다 (원본 사진은 저장소에 올리지 않는 것과 별개로, 이 실행에서만 전송됨).
"""
import argparse
import base64
import collections
import csv
import io
import json
import os
import random
import re
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
G00 = ROOT / "exp_grid_aug/outputs/g00_tile_list/tiles.csv"
OUT = HERE / "outputs/aa_caption_phrases"
PERSPECTIVES = [
    "Describe the overall impression of the sample surface in this microscope image in 1-2 sentences.",
    "Describe the texture and surface condition (smoothness, roughness, reflectivity) of the sample in 1-2 sentences.",
    "Describe any noticeable regions or irregularities on the sample surface in 1-2 sentences. If there are none, say so.",
]
SYSTEM = "You describe microscope images of metal sample surfaces for visual inspection. Describe only what is visible. Answer with the description only."
STOP = set("the a an of and or in on to is are with that this it its as by at from be there".split())


def select(seed=0, n=5):
    imgs = {}
    for r in csv.DictReader(open(G00, encoding="utf-8-sig")):
        imgs[r["image_id"]] = int(r["label"])
    rng = random.Random(seed); out = {}
    for lab in (0, 2):
        ids = sorted(i for i, l in imgs.items() if l == lab)
        rng.shuffle(ids); pick, metals = [], set()
        for i in ids:
            m = i.split("/")[0]
            if m not in metals:
                metals.add(m); pick.append(i)
            if len(pick) == n:
                break
        out[f"AGG{lab}"] = pick
    return out


def encode(path):
    im = Image.open(path).convert("RGB"); im.thumbnail((1568, 1568))
    b = io.BytesIO(); im.save(b, "JPEG", quality=92)
    return base64.standard_b64encode(b.getvalue()).decode()


def ngrams(text, lo=2, hi=4):
    w = re.findall(r"[a-z]+(?:-[a-z]+)*", text.lower()); out = set()
    for n in range(lo, hi + 1):
        for i in range(len(w) - n + 1):
            g = w[i:i + n]
            if g[0] in STOP or g[-1] in STOP:
                continue
            out.add(" ".join(g))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--model", default="claude-sonnet-5-5")
    a = ap.parse_args()
    sel = select()
    for lab, ids in sel.items():
        print(lab, ids)
    if a.dry_run:
        return
    import anthropic
    ws = os.environ.get("ANTHROPIC_WORKSPACE_ID")      # 키가 워크스페이스에 연결돼 있지 않을 때만 필요
    client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": ws} if ws else None)
    OUT.mkdir(parents=True, exist_ok=True)
    caps = []
    for lab, ids in sel.items():
        for iid in ids:
            data = encode(ROOT / (iid + ".jpg"))
            for k, q in enumerate(PERSPECTIVES, 1):
                msg = client.messages.create(model=a.model, max_tokens=200, system=SYSTEM, messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}, {"type": "text", "text": q}]}])
                caps.append(dict(label=lab, image_id=iid, perspective=k, caption=msg.content[0].text.strip()))
            print(" ", lab, iid, flush=True)
    json.dump(dict(model=a.model, captions=caps), open(OUT / "captions.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    cnt = {l: collections.Counter() for l in sel}
    ncap = {l: sum(1 for c in caps if c["label"] == l) for l in sel}
    for c in caps:
        cnt[c["label"]].update(ngrams(c["caption"]))
    allg = set().union(*[set(v) for v in cnt.values()])
    rows = [(g, cnt["AGG0"][g], cnt["AGG2"][g]) for g in allg if cnt["AGG0"][g] + cnt["AGG2"][g] >= 3]
    rows.sort(key=lambda r: -(r[1] + r[2]))
    with open(OUT / "phrases.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(["표현", f"AGG0 캡션 수 (전체 {ncap['AGG0']})", f"AGG2 캡션 수 (전체 {ncap['AGG2']})"]); w.writerows(rows)
    (OUT / "README.txt").write_text(__doc__.strip() + f"\n\n모델: {a.model}\n", encoding="utf-8")
    print("저장:", OUT)


if __name__ == "__main__":
    main()
