"""전처리 스크립트 공통 유틸: config 로딩, 로그, 파일명 파싱.

용어 (사용자 확인, 2026-10-12) — 코드/CSV 의 열 이름과 사람이 쓰는 말이 다르므로 주의
  1단계 폴더 (BASE, BASE-LC, BASE-LCSI, BASE-LSi, REF, REF-LN) = 금속 종류          [코드/CSV 열: treatment]
  2단계 폴더 (NPH1035, PH1180-1050 ...)                       = 처리 조건          [코드/CSV 열: sample]
      처리 조건 = 처리 방법(NPH, PH1180, PH1205, PH1230, PH1255) x 온도(1035, 1050)
  묶음(group) = 금속/처리 조건 폴더 = 같은 시편의 여러 배율 (60개)
  '시편' 은 현미경으로 찍은 파란 물체 자체를 가리킨다 (예: 시편 영역 마스크).
"""
import re
import sys
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PREPROC_DIR = Path(__file__).resolve().parent

# 정상 파일명:  x10#1 - AGG0.jpg / x12.5#1  - AGG1.jpg / X7.5#1 (수정함) - AGG2.jpg / x12.5 - AGG2.jpg (#번호 없음)
NAME_RE = re.compile(
    r"^[xX](?P<mag>\d+(?:\.\d+)?)(?:#(?P<rep>\d+))?(?P<fixed>\s*\(수정함\))?\s*-\s*AGG\s*(?P<label>[012])$"
)
# 알려진 오타 (s00a가 수정). 그 외 비정상 이름은 건드리지 않고 보고만 한다.
TYPO_HASH_BANG_RE = re.compile(r"^(?P<pre>[xX]\d+(?:\.\d+)?)#!(?P<post>\s*-\s*AGG\s*[012])$")
TYPO_NO_HASH_RE = re.compile(r"^(?P<pre>[xX]10)1(?P<post>\s*-\s*AGG\s*[012])$")


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    for k, v in cfg["paths"].items():
        cfg["paths"][k] = str((base / v).resolve())
    cfg["_config_path"] = str(path)
    return cfg


class Logger:
    """stdout + outputs/<script>/log.txt 동시 기록."""

    def __init__(self, out_dir, script_name):
        self.out_dir = Path(out_dir) / script_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.out_dir / "log.txt", "w", encoding="utf-8")
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg)
        self.fh.write(msg + "\n")
        self.fh.flush()

    def warn(self, *a):
        self("[WARN]", *a)


def list_images(root, exclude_dirs=()):
    """root 아래 이미지 파일 (정렬, 결정적). exclude_dirs 하위는 제외."""
    root = Path(root).resolve()
    ex = [Path(e).resolve() for e in exclude_dirs]
    out = []
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        if any(e in f.resolve().parents for e in ex):
            continue
        out.append(f)
    return out


def write_bytes_retry(path, buf, retries=8):
    """바이트 배열을 파일로 쓴다. Windows 에서 방금 열어본/전송한 PNG 를 같은 이름으로 덮어쓸 때 간헐적으로
    OSError(22)가 나서(원인은 확정하지 못함: 뷰어 등이 파일을 잡고 있는 것으로 추정) 짧게 대기하며 재시도한다."""
    import time
    for i in range(retries):
        try:
            buf.tofile(str(path)) if hasattr(buf, "tofile") else Path(path).write_bytes(buf)
            return
        except OSError:
            if i == retries - 1:
                raise RuntimeError(f"파일을 쓸 수 없음 (다른 프로그램에서 열려 있을 수 있음. 닫고 다시 실행): {path}")
            time.sleep(0.5 * (i + 1))


def imwrite_png(path, img):
    """OpenCV 이미지를 PNG 로 저장 (한글 경로 안전 + 재시도)."""
    import cv2
    write_bytes_retry(path, cv2.imencode(".png", img)[1])


def savefig_retry(fig, path, dpi=110):
    """matplotlib 그림을 메모리에서 PNG 로 만든 뒤 재시도하며 저장."""
    import io
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=dpi)
    write_bytes_retry(path, b.getvalue())


def farthest_point(xy, k, tile=224):
    """타일 좌상단 좌표 목록 xy 에서 k개를 서로 가장 멀리 떨어지게 고른다 (결정적).
    첫 타일은 타일 중심들의 무게중심에 가장 가까운 것, 이후 이미 고른 타일과의 최소 거리가 가장 큰 타일을 반복 선택.
    반환: 선택한 인덱스 목록 (선택 순서)."""
    import numpy as np
    c = np.array([(x + tile / 2, y + tile / 2) for x, y in xy], float)
    if len(c) <= k:
        return list(range(len(c)))
    first = int(np.argmin(((c - c.mean(0)) ** 2).sum(1)))
    chosen = [first]
    d = ((c - c[first]) ** 2).sum(1)
    while len(chosen) < k:
        j = int(np.argmax(d))
        chosen.append(j)
        d = np.minimum(d, ((c - c[j]) ** 2).sum(1))
    return chosen


def safe_id(image_id):
    """image_id('금속/처리 조건/파일명') -> 파일명으로 쓸 수 있는 문자열."""
    return image_id.replace("/", "__")


def parse_name(stem):
    """파일명(확장자 제외) -> dict 또는 None."""
    m = NAME_RE.match(stem)
    if not m:
        return None
    return {
        "mag": m.group("mag"),
        "rep": int(m.group("rep")) if m.group("rep") else None,
        "label": int(m.group("label")),
        "fixed": bool(m.group("fixed")),
    }
