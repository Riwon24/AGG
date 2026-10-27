"""aa_make_eval_script: AnomalyAgent 저장소의 evaluate_headCT_v12_target.py 를 복사해서 AGG 용 평가 스크립트 evaluate_agg_v12_target.py 를 만든다 (WSL ~/AnomalyAgent 안에 생성).

저자들의 평가 흐름(에이전트 실행, 점수 수집, AUROC 등 지표, 도구 호출 기록)은 그대로 두고, 아래만 바꾼다.
  - 데이터 로더: HeadCT labels.csv 대신 AGG 이미지(exp_grid_aug g00 의 이미지 목록)를 읽는다. 단위는 --unit image | tile.
      image : AGG0 vs AGG1+2 (이미지 라벨), 후보 문장을 만든 이미지는 기본으로 뺀다 (--include_candidate_sources 로 포함 가능)
      tile  : aa_caption_tiles 의 34개 타일 (AGG1 표시 17 = 이상, AGG0 대조 17 = 정상). 후보를 만든 타일이라 성능이 부풀려진다 (점검용).
  - 클래스 이름 "metal sample surface", 이상 정의(--target_anomaly_definition 으로 바꿀 수 있음).
  - --dry_run_data : 모델을 올리지 않고 데이터 구성만 출력하고 끝낸다.
  - --atomic_candidates_file 을 주면 exp_aa/outputs/aa_caption_tiles/ 의 같은 이름 파일을 save_path 로 복사해서 쓴다.
실행: python aa_make_eval_script.py  (그 뒤 ~/AnomalyAgent 에서 python evaluate_agg_v12_target.py ...)
"""
import os
import re
from pathlib import Path

SRC = Path(os.path.expanduser("~/AnomalyAgent/evaluate_headCT_v12_target.py"))
DST = SRC.with_name("evaluate_agg_v12_target.py")

LOADER = '''
AGG_DATA_ROOT = os.environ.get("AGG_DATA_ROOT", "/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")
AGG_CAPTION_DIR = Path(AGG_DATA_ROOT) / "exp_aa/outputs/aa_caption_tiles"
AGG_TILE = 224


def _agg_candidate_sources() -> Dict[str, List[Dict]]:
    """후보 문장을 만들 때 쓴 타일 34개 (aa_caption_tiles/captions.json)."""
    caps = json.load(open(AGG_CAPTION_DIR / "captions.json", encoding="utf-8"))["captions"]
    seen: Dict[int, Dict] = {}
    for c in caps:
        seen.setdefault(c["n"], c)
    return list(seen.values())


def load_agg_test_data(args) -> List[Dict]:
    """AGG 평가 데이터. --unit image: 이미지 단위 AGG0 vs AGG1+2, --unit tile: 후보를 만든 34개 타일."""
    root = Path(args.data_path)
    sources = _agg_candidate_sources()
    test_data: List[Dict] = []
    if args.unit == "tile":
        crop_dir = Path(args.save_path) / "tiles"
        crop_dir.mkdir(parents=True, exist_ok=True)
        tiles = {}
        for r in csv.DictReader(open(root / "exp_grid_aug/outputs/g00_tile_list/tiles.csv", encoding="utf-8-sig")):
            tiles[(r["image_id"], r["tile_id"])] = r
        for c in sources:
            r = tiles[(c["image_id"], c["tile_id"])]
            x0, y0 = int(r["x0"]), int(r["y0"])
            im = Image.open(root / (c["image_id"] + ".jpg")).convert("RGB").crop((x0, y0, x0 + AGG_TILE, y0 + AGG_TILE)).resize((448, 448), Image.LANCZOS)
            p = crop_dir / f"{c['label']}_{c['n']:02d}.png"
            im.save(p)
            test_data.append({"full_img_path": str(p), "full_mask_path": None, "img_path": p.name, "cls_name": AGG_CLASS_NAME,
                              "anomaly": 1 if c["label"] == "AGG1표시" else 0, "id": f"{c['image_id']}|{c['tile_id']}", "label_value": c["label"]})
        return test_data
    exclude = set() if args.include_candidate_sources else {c["image_id"] for c in sources}
    labels = {}
    for r in csv.DictReader(open(root / "exp_grid_aug/outputs/g00_tile_list/tiles.csv", encoding="utf-8-sig")):
        labels[r["image_id"]] = int(r["label"])
    for iid in sorted(labels):
        if iid in exclude:
            continue
        p = root / (iid + ".jpg")
        test_data.append({"full_img_path": str(p), "full_mask_path": None, "img_path": iid + ".jpg", "cls_name": AGG_CLASS_NAME,
                          "anomaly": 1 if labels[iid] > 0 else 0, "id": iid, "label_value": f"AGG{labels[iid]}"})
    print(f"후보 문장을 만든 이미지 {len(exclude)}장 제외" if exclude else "후보 문장을 만든 이미지도 포함 (성능이 부풀려질 수 있음)")
    return test_data

'''

ARGS = '''    parser.add_argument("--unit", choices=["image", "tile"], default="image", help="AGG 평가 단위: image(이미지 단위 AGG0 vs AGG1+2) 또는 tile(후보를 만든 34개 타일, 점검용)")
    parser.add_argument("--include_candidate_sources", action="store_true", help="후보 문장을 만든 이미지도 평가에 포함 (기본은 제외)")
    parser.add_argument("--dry_run_data", action="store_true", help="모델을 올리지 않고 데이터 구성만 출력하고 종료")

'''

DEFINITION = '''AGG_CLASS_NAME = "metal sample surface"
AGG_TARGET_ANOMALY_DEFINITION = (
    "Only a flat, smooth region that differs from the surrounding fine granular texture of the sample surface is anomalous (AGG). "
    "The dark background, sample edges and boundaries, lighting bands or glare, scratches, fibers, dust, and stains are not anomalous; "
    "a sample without such a flat, smooth region should be judged normal."
)
'''


def sub1(pattern, repl, text, flags=0):
    new, n = re.subn(pattern, lambda m: repl, text, count=1, flags=flags)
    assert n == 1, f"패턴을 찾지 못함: {pattern[:50]}"
    return new


def main():
    s = SRC.read_text(encoding="utf-8")
    s = sub1(r'DEFAULT_HEADCT_DATA_PATH = [^\n]*\n', 'DEFAULT_HEADCT_DATA_PATH = os.environ.get("AGG_DATA_ROOT", "/mnt/c/Users/admin/Downloads/260929_images_for_AGG - manual labeling/AGG_labeled_only")\n', s)
    s = sub1(r'HEADCT_CLASS_NAME = "brain CT image"\nHEADCT_TARGET_ANOMALY_DEFINITION = \(.*?\n\)\n', 'HEADCT_CLASS_NAME = "metal sample surface"\n' + DEFINITION.split("\n", 1)[1].replace("AGG_TARGET_ANOMALY_DEFINITION", "HEADCT_TARGET_ANOMALY_DEFINITION"), s, flags=re.S)
    s = sub1(r'\ndef limit_samples_balanced', "\nAGG_CLASS_NAME = HEADCT_CLASS_NAME\n" + LOADER + "\ndef limit_samples_balanced", s)
    s = sub1(r'    test_data = load_headct_test_data\(args\.data_path, args\.labels_csv\)\n', '    test_data = load_agg_test_data(args)\n', s)
    s = sub1(r'    print\(f"Loading HeadCT labels from \{args\.labels_csv\}\.\.\."\)\n', '', s)
    s = sub1(r'    args = parser\.parse_args\(\)\n', ARGS + '    args = parser.parse_args()\n', s)
    s = sub1(r'    os\.makedirs\(args\.save_path, exist_ok=True\)\n',
             '    os.makedirs(args.save_path, exist_ok=True)\n'
             '    if args.atomic_candidates_file:\n'
             '        import shutil\n'
             '        _dst = os.path.join(args.save_path, args.atomic_candidates_file)\n'
             '        _src = os.path.join(str(AGG_CAPTION_DIR), args.atomic_candidates_file)\n'
             '        if not os.path.exists(_dst) and os.path.isfile(_src):\n'
             '            shutil.copy(_src, _dst)\n'
             '            print(f"후보 파일 복사: {_src} -> {_dst}")\n', s)
    s = sub1(r'(    print\(f"Label counts: normal=\{normal_count\}, anomaly=\{anomaly_count\}"\)\n)',
             r'\1' if False else '    print(f"Label counts: normal={normal_count}, anomaly={anomaly_count}")\n    if args.dry_run_data:\n        print("--dry_run_data: 데이터 구성만 확인하고 종료합니다.")\n        return\n', s)
    s = s.replace('"./results/headCT_agent_v1_16_target_eval"', '"./results/agg_agent_v1_16_target_eval"')
    DST.write_text(s, encoding="utf-8")
    print("생성:", DST)


if __name__ == "__main__":
    main()
