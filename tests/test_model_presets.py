import unittest

from open_storyline.model_presets import (
    ATLAS_CLOUD_MODEL_KEY,
    list_model_presets,
    resolve_model_preset,
)


class ModelPresetTests(unittest.TestCase):
    def test_atlas_cloud_is_available_for_llm_only(self):
        self.assertEqual(list_model_presets("llm"), [ATLAS_CLOUD_MODEL_KEY])
        self.assertEqual(list_model_presets("vlm"), [])

    def test_atlas_cloud_resolves_openai_compatible_config(self):
        config, error = resolve_model_preset(
            "llm",
            ATLAS_CLOUD_MODEL_KEY,
            {"ATLASCLOUD_API_KEY": "test-key"},
        )

        self.assertIsNone(error)
        self.assertEqual(
            config,
            {
                "model": "deepseek-ai/deepseek-v4-pro",
                "base_url": "https://api.atlascloud.ai/v1",
                "api_key": "test-key",
            },
        )

    def test_atlas_cloud_accepts_legacy_env_name(self):
        config, error = resolve_model_preset(
            "llm",
            ATLAS_CLOUD_MODEL_KEY,
            {"ATLAS_CLOUD_API_KEY": "legacy-key"},
        )

        self.assertIsNone(error)
        self.assertEqual(config["api_key"], "legacy-key")

    def test_atlas_cloud_reports_missing_api_key(self):
        config, error = resolve_model_preset("llm", ATLAS_CLOUD_MODEL_KEY, {})

        self.assertIsNone(config)
        self.assertIn("ATLASCLOUD_API_KEY", error)
        self.assertIn("ATLAS_CLOUD_API_KEY", error)


if __name__ == "__main__":
    unittest.main()
