"""MIL 실험 공통: config, 이진 라벨 이미지 표, 분할, 타일 특징(버전별) 로딩, gated attention MIL 모델과 학습 함수.

exp_group_cv/gcommon.py (이미지 표, 그룹 분할, 증강 특징 로딩)와 exp_binary/bcommon.py (이진 지표)를 읽기 전용으로 불러 쓴다. 분할과 특징 추출 파이프라인은 바꾸지 않는다.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "exp_binary"))
sys.path.insert(0, str(HERE.parent / "exp_group_cv"))                   # bcommon 이 self-contained 로 바뀌어 gcommon 의 경로를 더 이상 넣어 주지 않는다
from bcommon import metrics_bin  # noqa: E402,F401
from gcommon import Logger, check_folds, load_features, load_images, make_folds, write_csv  # noqa: E402,F401
from gcommon import load_config as _load_config  # noqa: E402


def load_config(path=None):
    return _load_config(path or HERE / "config.yaml")


def to_binary(y3, cfg):
    return np.isin(y3, cfg["positive_labels"]).astype(int)


class GatedAttentionMIL(nn.Module):
    """Ilse, Tomczak, Welling (2018) 의 gated attention MIL: 타일 임베딩 h -> 어텐션 a_k = softmax_k( w^T( tanh(V h_k) * sigmoid(U h_k) ) ) -> bag = sum_k a_k h_k -> 분류기."""

    def __init__(self, d_in, embed_dim, attn_dim, p):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_in, embed_dim), nn.ReLU(), nn.Dropout(p))
        self.V = nn.Linear(embed_dim, attn_dim)
        self.U = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(embed_dim, 1))

    def forward(self, X, mask):
        H = self.embed(X)                                                              # (B, N, E)
        A = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))).squeeze(-1)       # (B, N)
        A = A.masked_fill(~mask, float("-inf"))
        a = torch.softmax(A, dim=1)                                                    # 패딩은 0
        M = (a.unsqueeze(-1) * H).sum(1)                                               # (B, E) bag 표현
        return self.head(M).squeeze(-1), a


class AdditiveGatedAttentionMIL(nn.Module):
    """GatedAttentionMIL 의 pooling 과 classifier 의 합성 순서를 바꾼 additive 판 (Javed et al. 2022, Additive MIL).
    embed 와 attention 은 GatedAttentionMIL 과 같은 구조이고, head(분류기)를 pooling 전에 타일마다 적용한다.
      (기존)     logit = head( sum_k a_k h_k )
      (additive) logit = sum_k a_k z_k + b,  z_k = head(h_k)   -> a_k z_k 가 그 타일이 bag logit 에 더한 기여도이고, 기여도의 합 + b 가 bag logit 과 정확히 같다.
    타일별 head 는 bias 가 없고 bag 수준 bias b 하나만 둔다. 반환: (bag logit, attention a, 타일별 기여도 a_k z_k)."""

    def __init__(self, d_in, embed_dim, attn_dim, p):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(d_in, embed_dim), nn.ReLU(), nn.Dropout(p))
        self.V = nn.Linear(embed_dim, attn_dim)
        self.U = nn.Linear(embed_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.head = nn.Sequential(nn.Dropout(p), nn.Linear(embed_dim, 1, bias=False))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X, mask):
        H = self.embed(X)                                                              # (B, N, E)
        A = self.w(torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))).squeeze(-1)       # (B, N)
        A = A.masked_fill(~mask, float("-inf"))
        a = torch.softmax(A, dim=1)                                                    # 패딩은 0
        z = self.head(H).squeeze(-1)                                                   # (B, N) 타일별 logit
        contrib = a * z                                                                # (B, N) 타일별 기여도 (패딩은 0)
        logit = contrib.masked_fill(~mask, 0.0).sum(1) + self.bias                     # (B,)
        return logit, a, contrib


def pad_bags(bags, nmax, d, device):
    """bag 목록 [(n_i, d) float32] -> (B, nmax, d) 텐서와 (B, nmax) bool 마스크 (nmax 보다 짧은 bag 은 0 패딩)"""
    X = np.zeros((len(bags), nmax, d), np.float32); M = np.zeros((len(bags), nmax), bool)
    for j, b in enumerate(bags):
        X[j, :len(b)] = b; M[j, :len(b)] = True
    return torch.from_numpy(X).to(device), torch.from_numpy(M).to(device)


def standardize_fit(train_bags):
    """학습 bag(모든 버전)의 타일만으로 평균과 표준편차를 구한다 (test 타일은 쓰지 않는다)."""
    Z = np.concatenate(train_bags, 0)
    return Z.mean(0).astype(np.float32), (Z.std(0) + 1e-6).astype(np.float32)


def train_mil(train_bags, ytr, cfg, seed, device, nmax):
    """학습 bag(이미 표준화됨)로 MIL 을 학습한다. 고정 epoch, 조기 종료 없음. 반환: (모델, 마지막 epoch 평균 손실, 학습 bag accuracy)"""
    mc = cfg["mil"]
    torch.manual_seed(seed); np.random.seed(seed)
    d = train_bags[0].shape[1]
    mtype = mc.get("model_type", "gated")                                       # gated = 기존 (기본), additive = 타일별 기여도 판
    assert mtype in ("gated", "additive"), mtype
    cls_ = GatedAttentionMIL if mtype == "gated" else AdditiveGatedAttentionMIL
    model = cls_(d, mc["embed_dim"], mc["attn_dim"], mc["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    X, M = pad_bags(train_bags, nmax, d, device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    n1 = float((ytr == 1).sum()); n0 = float((ytr == 0).sum())
    w = torch.where(yt == 1, torch.tensor(len(ytr) / (2 * n1), device=device), torch.tensor(len(ytr) / (2 * n0), device=device)) if mc["class_balanced_loss"] else torch.ones_like(yt)
    g = torch.Generator(device="cpu"); g.manual_seed(seed)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    model.train()
    last = 0.0
    for ep in range(mc["epochs"]):
        perm = torch.randperm(len(ytr), generator=g).to(device); tot = 0.0
        for i in range(0, len(ytr), mc["batch_bags"]):
            idx = perm[i:i + mc["batch_bags"]]
            logit = model(X[idx], M[idx])[0]
            loss = (w[idx] * bce(logit, yt[idx])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
        last = tot / len(ytr)
    model.eval()
    with torch.no_grad():
        logit = model(X, M)[0]
        acc = float(((logit > 0).float() == yt).float().mean())
    return model, last, acc


@torch.no_grad()
def predict_mil(model, bags, nmax, device):
    """bag 목록 -> (확률 (B,), attention 목록 [(n_i,)], 기여도 목록 [(n_i,)]). 기여도는 additive 모델에서만 있고 gated 모델이면 None 목록이다."""
    d = bags[0].shape[1]
    X, M = pad_bags(bags, nmax, d, device)
    out = model(X, M)
    p = torch.sigmoid(out[0]).cpu().numpy(); a = out[1].cpu().numpy()
    if len(out) > 2:
        c = out[2].cpu().numpy(); cons = [c[j, :len(b)] for j, b in enumerate(bags)]
    else:
        cons = [None] * len(bags)
    return p, [a[j, :len(b)] for j, b in enumerate(bags)], cons


def tile_keep(cfg, tiles_of):
    """config tile_filter: all = 모든 분석 타일, full = 마스크 100% 타일(시편만 있는 타일)만, min_frac = 마스크 비율 >= min_mask_frac 인 타일만. 반환: 이미지별로 tiles_of[i] 안에서 남길 타일의 위치 배열.
    full 모드의 타일 특징은 배경 증강 버전 사이에 픽셀이 같다 (시편만 있으므로). 이 사실은 호출한 쪽에서 assert 로 확인한다."""
    mode = cfg.get("tile_filter", "all")
    assert mode in ("all", "full", "min_frac"), mode
    if mode == "all":
        return [np.arange(len(t)) for t in tiles_of]
    tindex = list(csv.DictReader(open(f"{cfg['paths']['aug_out']}/e01_extract_features/tile_index.csv", encoding="utf-8-sig")))
    mf = {(r["image_id"], r["tile_id"]): float(r["mask_frac"]) for r in csv.DictReader(open(cfg["paths"]["tiles_csv"], encoding="utf-8-sig"))}
    thr = 0.999999 if mode == "full" else float(cfg["min_mask_frac"])      # min_frac: 마스크 비율(시편 픽셀 비율)이 config min_mask_frac 이상인 타일만 (시편 비율이 낮은 가장자리 타일 제외)
    return [np.array([j for j, ti in enumerate(t) if mf[(tindex[ti]["image_id"], tindex[ti]["tile_id"])] >= thr], dtype=int) for t in tiles_of]
