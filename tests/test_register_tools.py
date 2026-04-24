from open_storyline.codex.sampling_handler import CodexSamplingLLMClient
from open_storyline.mcp.register_tools import _build_node_llm_from_params


def test_build_node_llm_from_params_uses_codex_direct_sampling() -> None:
    params = {
        "_storyline_sampling_backend": "codex",
        "_storyline_sampling_model": "gpt-5.4",
        "_storyline_sampling_reasoning_effort": "high",
    }

    llm = _build_node_llm_from_params(object(), params)

    assert isinstance(llm, CodexSamplingLLMClient)
    assert "_storyline_sampling_backend" not in params
    assert "_storyline_sampling_model" not in params
    assert "_storyline_sampling_reasoning_effort" not in params
