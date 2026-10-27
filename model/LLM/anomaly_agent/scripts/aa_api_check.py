"""aa_api_check: 에이전트가 쓰는 경로(react_agent.context._load_api_model)로 Claude 를 짧게 한 번 호출해서, 키와 워크스페이스 ID 설정이 맞는지 확인한다.
필요: 환경변수 ANTHROPIC_API_KEY (+ 키가 워크스페이스에 연결돼 있지 않으면 ANTHROPIC_WORKSPACE_ID). 호출은 한 번뿐이다.
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~/AnomalyAgent/src"))
from react_agent.context import _load_api_model  # noqa: E402

print("키 설정:", "있음" if os.environ.get("ANTHROPIC_API_KEY") else "없음", "| 워크스페이스 ID 설정:", "있음" if os.environ.get("ANTHROPIC_WORKSPACE_ID") else "없음")
model = _load_api_model("anthropic/claude-sonnet-5-5")
print("응답:", model.invoke("Reply with the single word: ok").content)
