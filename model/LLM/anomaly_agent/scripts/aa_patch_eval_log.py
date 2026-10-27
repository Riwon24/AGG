"""aa_patch_eval_log: 에이전트가 사진마다 계산한 연속 점수(반사실 margin, 일반 템플릿 점수)를 results.json 에 남기도록 고친다.
  tools.py              : 모듈 변수 RUN_LOG 에 일반 템플릿 / 반사실 도구의 점수를 기록
  evaluate_agg_v12_target.py : 사진마다 RUN_LOG 를 비우고 돌린 뒤, 결과 항목에 "template_log" 로 저장
이미 적용돼 있으면 건너뛴다. 실행: python aa_patch_eval_log.py (WSL)
"""
import os

ROOT = os.path.expanduser("~/AnomalyAgent")


def patch(path, pairs, marker):
    s = open(path, encoding="utf-8", newline="").read()
    if marker in s:
        print(os.path.basename(path), "이미 적용됨")
        return
    for old, new, cnt in pairs:
        assert s.count(old) == cnt, (os.path.basename(path), old[:50], s.count(old))
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="").write(s)
    print(os.path.basename(path), "수정 완료")


patch(os.path.join(ROOT, "src/react_agent/tools.py"), [
    ("def _sigmoid(x: float) -> float:",
     "# 평가 스크립트가 사진마다 읽어 가는 연속 점수 기록 (results.json 의 template_log)\nRUN_LOG: Dict[str, Any] = {}\n\n\ndef _sigmoid(x: float) -> float:", 1),
    ('    avg = float(np.mean(scores))\n    label = "anomalous" if avg > 0.5 else "normal"\n',
     '    avg = float(np.mean(scores))\n    label = "anomalous" if avg > 0.5 else "normal"\n    if not decision:\n        RUN_LOG["general"] = {"margins": margins, "scores": scores, "avg_score": avg, "avg_margin": float(np.mean(margins))}\n', 1),
    ('    _state_set(state, "counterfactual_stats", stats)\n',
     '    _state_set(state, "counterfactual_stats", stats)\n    RUN_LOG["counterfactual"] = {**stats, "margins": [p["margin"] for p in per]}\n', 1),
], "RUN_LOG")

patch(os.path.join(ROOT, "evaluate_agg_v12_target.py"), [
    ("            result_state = await graph_instance.ainvoke(initial_state, config=config, context=context)\n",
     "            import react_agent.tools as _aa_tools\n            _aa_tools.RUN_LOG.clear()\n            result_state = await graph_instance.ainvoke(initial_state, config=config, context=context)\n            _AA_LOGS[image_path] = copy.deepcopy(_aa_tools.RUN_LOG)\n", 1),
    ('                    "label_value": sample["label_value"],\n', '                    "label_value": sample["label_value"],\n                    "template_log": _AA_LOGS.get(sample["full_img_path"]),\n', 2),
    ("\ndef limit_samples_balanced", "\nimport copy\n_AA_LOGS: Dict[str, Any] = {}\n\n\ndef limit_samples_balanced", 1),
], "_AA_LOGS")
