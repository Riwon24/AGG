"""aa_patch_system_merge: Anthropic API 는 system 메시지가 한 덩어리로 맨 앞에 있어야 한다.
planner 의 메시지 목록은 [system, (사용자 입력), system(템플릿 분석), user] 순서라서 langchain 이
"Received multiple non-consecutive system messages" 로 호출을 거부한다. 그래서 safe_api_invoke 가 anthropic 일 때
모든 system 메시지를 하나로 합쳐 맨 앞에 두도록 고친다. 이미 적용돼 있으면 건너뛴다. 실행: python aa_patch_system_merge.py (WSL)
"""
import os

P = os.path.expanduser("~/AnomalyAgent/src/react_agent/utils.py")
s = open(P, encoding="utf-8", newline="").read()
if "_merge_system_messages" in s:
    print("이미 적용됨")
    raise SystemExit

HELPER = r'''def _merge_system_messages(messages: List[BaseMessage | Dict[str, str]]) -> List[BaseMessage | Dict[str, str]]:
    """Merge every system message into one SystemMessage at the front (Anthropic rejects non-consecutive system messages)."""
    from langchain_core.messages import SystemMessage

    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(it.get("text", "") if isinstance(it, dict) else str(it) for it in content)
        return str(content)

    system_texts: List[str] = []
    rest: List[BaseMessage | Dict[str, str]] = []
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "system":
            system_texts.append(_text(m.get("content", "")))
        elif isinstance(m, SystemMessage):
            system_texts.append(_text(m.content))
        else:
            rest.append(m)
    if not system_texts:
        return messages
    return [SystemMessage(content="\n\n".join(t for t in system_texts if t)), *rest]


async def safe_api_invoke('''
assert s.count("async def safe_api_invoke(") == 1
s = s.replace("async def safe_api_invoke(", HELPER, 1)
OLD = "        api_provider = get_api_model_provider(api_model_id)\n"
assert s.count(OLD) >= 1
i = s.index("async def safe_api_invoke") if False else s.index(OLD)
s = s[:i] + OLD + '        if api_provider == "anthropic":\n            normalized_messages = _merge_system_messages(normalized_messages)\n' + s[i + len(OLD):]
open(P, "w", encoding="utf-8", newline="").write(s)
print("수정 완료")
