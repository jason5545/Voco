import importlib.util
import json
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "generate_localization.py"
SPEC = importlib.util.spec_from_file_location("generate_localization", MODULE_PATH)
localization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(localization)


def localized(value):
    return localization.string_localization(value)


class LocalizationGeneratorTests(unittest.TestCase):
    def test_bootstrap_maps_brand_and_preserves_fork_translation(self):
        legacy = {
            "strings": {
                "Dismiss Voco Recorder": {
                    "localizations": {"zh-Hant": localized("關閉 Voco 錄音")}
                }
            }
        }
        upstream = {"strings": {"Dismiss VoiceInk Recorder": {}}}
        overrides = localization.bootstrap_overrides(legacy, upstream)
        entry = overrides["strings"]["Dismiss VoiceInk Recorder"]
        self.assertEqual(localization.localization_value(entry, "en"), "Dismiss Voco Recorder")
        self.assertEqual(localization.localization_value(entry, "zh-Hant"), "關閉 Voco 錄音")
        self.assertNotIn("Dismiss Voco Recorder", overrides["strings"])

    def test_bootstrap_keeps_alias_for_upstream_deleted_view(self):
        legacy = {
            "strings": {
                "Voco is private": {
                    "localizations": {"zh-Hant": localized("Voco 預設保護隱私")}
                }
            }
        }
        overrides = localization.bootstrap_overrides(legacy, {"strings": {}})
        self.assertIn("Voco is private", overrides["strings"])
        self.assertIn("VoiceInk is private", overrides["strings"])
        self.assertEqual(
            localization.localization_value(overrides["strings"]["VoiceInk is private"], "en"),
            "Voco is private",
        )

    def test_generate_preserves_upstream_languages(self):
        upstream = {
            "sourceLanguage": "en",
            "strings": {
                "Hello": {
                    "localizations": {
                        "de": localized("Hallo"),
                        "fr": localized("Bonjour"),
                        "zh-Hans": localized("你好"),
                    }
                }
            },
        }
        overrides = {"strings": {"Hello": {"localizations": {"zh-Hant": localized("你好")}}}}
        catalog = localization.generate_catalog(upstream, overrides)
        self.assertEqual(set(catalog["strings"]["Hello"]["localizations"]), {"de", "fr", "zh-Hans", "zh-Hant"})

    def test_plural_variations_are_preserved_and_validated(self):
        upstream = {
            "sourceLanguage": "en",
            "strings": {
                "%lld items": {
                    "localizations": {
                        "en": {
                            "variations": {
                                "plural": {
                                    "one": localized("%lld item"),
                                    "other": localized("%lld items"),
                                }
                            }
                        }
                    }
                }
            },
        }
        overrides = {
            "strings": {
                "%lld items": {
                    "localizations": {
                        "zh-Hant": {
                            "variations": {"plural": {"other": localized("%lld 個項目")}}
                        }
                    }
                }
            }
        }
        catalog = localization.generate_catalog(upstream, overrides)
        self.assertEqual(localization.localization_values(catalog["strings"]["%lld items"], "en"), ["%lld item", "%lld items"])
        self.assertEqual(localization.localization_values(catalog["strings"]["%lld items"], "zh-Hant"), ["%lld 個項目"])

    def test_positional_and_unpositional_formats_are_compatible(self):
        self.assertEqual(
            localization.format_specifiers("%1$@ → %2$lld"),
            localization.format_specifiers("%@ → %lld"),
        )

    def test_plural_suffix_may_be_omitted_but_number_type_may_not_change(self):
        localization.validate_catalog({
            "sourceLanguage": "en",
            "strings": {
                "%lld file%@": {
                    "localizations": {"zh-Hant": localized("%1$lld 個檔案")}
                }
            },
        })
        catalog = {
            "sourceLanguage": "en",
            "strings": {"Count %lld": {"localizations": {"zh-Hant": localized("數量 %@")}}},
        }
        with self.assertRaisesRegex(ValueError, "format specifiers"):
            localization.validate_catalog(catalog)

    def test_duplicate_json_key_is_rejected(self):
        path = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "duplicate.json"
        path.write_text('{"strings": {}, "strings": {}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            localization.load_json(path)


if __name__ == "__main__":
    unittest.main()
