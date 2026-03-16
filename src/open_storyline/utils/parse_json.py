import json
import re
from typing import Any, Dict, Optional, Iterable, List

def try_parse_tool_call(text:str) -> Optional[Dict[str, Any]]:
    """
    Return dict if text is a valid tool call JSON, otherwise return None
    """
    try:
        obj = parse_json_dict(text)
    except:
        return None
    
    if obj.get("action") != "call_tool":
        return None
    if "tool" not in obj:
        return None
    
    args = obj.get("arguments", {})
    if args is not None and not isinstance(args, dict):
        return None
    
    return obj

# Support ```json ... ``` and ```jsonc ... ``` (can remove jsonc if needed)
_CODE_FENCE_RE = re.compile(
    r"```(?:json|jsonc)\s*(.*?)\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)


def _strip_trailing_commas_once(s: str) -> str:
    """
    Remove trailing commas before '}' or ']' in JSON text (single pass).
    Note: Skips content inside strings, won't remove commas within strings.
    """
    out = []
    in_str = False
    escape = False
    i = 0
    n = len(s)

    while i < n:
        c = s[i]

        if in_str:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            i += 1
            continue

        # not in string
        if c == '"':
            in_str = True
            out.append(c)
            i += 1
            continue

        if c == ",":
            # look ahead to next non-whitespace
            j = i + 1
            while j < n and s[j] in " \t\r\n":
                j += 1
            if j < n and s[j] in "}]":
                # drop this comma
                i += 1
                continue

        out.append(c)
        i += 1

    return "".join(out)


def _strip_trailing_commas(s: str, max_passes: int = 10) -> str:
    """
    Remove extra commas before '}' or ']' in JSON text (single pass).
    Note: String content is skipped, so commas inside strings won't be removed.
    """
    for _ in range(max_passes):
        s2 = _strip_trailing_commas_once(s)
        if s2 == s:
            return s2
        s = s2
    return s  # best effort


def _extract_balanced_object(text: str, start: int) -> Optional[str]:
    """
    Extract a balanced JSON object substring {...} starting from text[start] == '{'.
    Correctly skips braces within strings.
    """
    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]

        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue

        if c == '"':
            in_str = True
            continue

        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _iter_fenced_json_blocks(text: str) -> Iterable[str]:
    for m in _CODE_FENCE_RE.finditer(text):
        block = m.group(1)
        if block is not None:
            yield block.strip()


def _iter_object_candidates(text: str) -> Iterable[str]:
    """
    Enumerate all possible {...} substrings in arbitrary text (in order of appearance).
    """
    for idx, ch in enumerate(text):
        if ch == "{":
            cand = _extract_balanced_object(text, idx)
            if cand:
                yield cand

def parse_json_list(text: str) -> List[Any]:
    """
    Parse a JSON array (list) from arbitrary text.

    Supports:
      1) Markdown fenced JSON code blocks: ```json ... ```
      2) JSON surrounded by extra text
      3) Removing trailing commas before '}' or ']' 
    Args:
        text: Input string to parse
    Returns:
        Parsed list
    Raises:
        ValueError: Cannot find a valid JSON array to parse
        TypeError: Input text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # Try fenced block first, then try the entire text
    search_spaces = list(_iter_fenced_json_blocks(text))
    search_spaces.append(text)

    last_err: Optional[Exception] = None

    for space in search_spaces:
        stripped = space.lstrip().lstrip("\ufeff")  # 顺便去 BOM
        if not stripped.startswith("["):
            continue

        cleaned = _strip_trailing_commas(stripped).strip()
        try:
            obj = json.loads(cleaned)
            if isinstance(obj, list):
                return obj
        except Exception as e:
            last_err = e
            continue

    raise ValueError("No valid JSON array (list) found in input") from last_err


def parse_json_dict(text: str) -> Dict[str, Any]:
    """
    Parse a JSON object (dict) from arbitrary text.

    Supports:
      1) Markdown fenced JSON code blocks: ```json ... ```
      2) JSON surrounded by extra text
      3) Removing trailing commas before '}' or ']'

    Args:
        text: Input string to parse
    Returns:
        Parsed dictionary
    Raises:
        ValueError: Cannot find a valid JSON dict to parse
        TypeError: Input text is not a string
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # Try fenced block first, then try the entire text
    search_spaces = list(_iter_fenced_json_blocks(text))
    search_spaces.append(text)

    last_err: Optional[Exception] = None

    for space in search_spaces:
        # If starts with '{', try to extract a balanced object from the beginning first (to avoid trailing noise)
        candidates = []
        stripped = space.lstrip().lstrip("\ufeff")  # 顺便去 BOM
        if stripped.startswith("{"):
            first = _extract_balanced_object(stripped, 0)
            if first:
                candidates.append(first)

        # Also try objects appearing at any position in the text
        candidates.extend(_iter_object_candidates(space))

        # Deduplicate (avoid retrying same substrings)
        seen = set()
        for cand in candidates:
            if cand in seen:
                continue
            seen.add(cand)

            cleaned = _strip_trailing_commas(cand).strip()
            try:
                obj = json.loads(cleaned)
                if isinstance(obj, dict):
                    return obj
            except Exception as e:
                last_err = e
                continue

    raise ValueError("No valid JSON object (dict) found in input") from last_err