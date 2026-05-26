"""Extract runnable Python source from an LLM response."""

import re

_CODE_FENCE = re.compile(
    r"```[ \t]*(?:python|py)?[ \t]*\r?\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)


def extract_code(response: str) -> str:
    """Return the Python source contained in an LLM response.

    Prefers fenced code blocks (```python ... ```), returning the **last**
    one — chain-of-thought responses place reasoning first and the final
    implementation last. If no fenced block is present, the whole response
    is returned stripped (some models reply with bare code).
    """
    blocks = _CODE_FENCE.findall(response)
    if blocks:
        return blocks[-1].strip()
    return response.strip()
