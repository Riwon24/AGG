"""aa_patch_workspace_header: 워크스페이스에 연결되지 않은 Anthropic API 키는 요청마다 anthropic-workspace-id 헤더가 필요하다.
환경변수 ANTHROPIC_WORKSPACE_ID 가 있으면 react_agent/context.py, context_v1_27.py 의 _load_api_model 이 그 헤더를 보내도록 고친다. 이미 적용돼 있으면 건너뛴다.
실행 (WSL): python aa_patch_workspace_header.py
"""
import os

OLD = '''    if provider == "openai":
        init_kwargs["use_responses_api"] = use_responses_api
'''
NEW = '''    if provider == "openai":
        init_kwargs["use_responses_api"] = use_responses_api
    if provider == "anthropic" and os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        # 워크스페이스에 연결되지 않은 API 키는 요청마다 워크스페이스 ID 헤더가 필요하다
        init_kwargs["default_headers"] = {"anthropic-workspace-id": os.environ["ANTHROPIC_WORKSPACE_ID"]}
'''
for name in ("context.py", "context_v1_27.py"):
    p = os.path.expanduser(f"~/AnomalyAgent/src/react_agent/{name}")
    s = open(p, encoding="utf-8", newline="").read()
    if "ANTHROPIC_WORKSPACE_ID" in s:
        print(name, "이미 적용됨"); continue
    assert s.count(OLD) == 1, name
    assert "\nimport os" in s or "\nimport os\n" in s, name + ": os import 없음"
    open(p, "w", encoding="utf-8", newline="").write(s.replace(OLD, NEW))
    print(name, "수정 완료")
