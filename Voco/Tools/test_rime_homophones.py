#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR / "RimeHomophones"))

import common
import generate
import layer
import librime_check


class RimeHomophoneMigrationTests(unittest.TestCase):
    def test_profile_falls_back_to_terra_and_selects_penglai_userdb(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "terra_pinyin.userdb").mkdir()
            (root / "terra_pinyin.dict.yaml").write_text(
                "---\nname: terra_pinyin\n...\n測\tce4\t90%\n",
                encoding="utf-8",
            )
            (root / "penglai_zhuyin.dict.yaml").write_text(
                "---\nname: penglai_zhuyin\n...\n測\tㄘㄜˋ\t90%\n",
                encoding="utf-8",
            )

            old_profile = common.active_rime_profile(root)
            self.assertEqual(("terra_pinyin", "pinyin"), (old_profile.schema, old_profile.code_format))
            self.assertEqual({"ce4": 90.0}, common.load_char_readings(root, old_profile)["測"])

            (root / "penglai_zhuyin.userdb").mkdir()
            new_profile = common.active_rime_profile(root)
            self.assertEqual(("penglai_zhuyin", "zhuyin"), (new_profile.schema, new_profile.code_format))
            self.assertEqual({"ㄘㄜˋ": 90.0}, common.load_char_readings(root, new_profile)["測"])

    def test_learning_export_uses_the_selected_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "penglai_zhuyin.userdb").mkdir()
            profile = common.active_rime_profile(root)
            with patch.object(generate.subprocess, "run") as run:
                generate.export_learning_records(Path("/rime_dict_manager"), root, root / "out.tsv", profile)
            self.assertEqual("penglai_zhuyin", run.call_args.args[0][2])

    def test_keys_for_supports_pinyin_zhuyin_tones_and_fake_english_syllables(self):
        pinyin = common.RimeProfile(schema="terra_pinyin", code_format="pinyin")
        zhuyin = common.RimeProfile(schema="penglai_zhuyin", code_format="zhuyin")

        self.assertEqual("g~", librime_check.keys_for(["S~"], pinyin))
        self.assertEqual("hk4g4", librime_check.keys_for(["ㄘㄜˋ", "ㄕˋ"], zhuyin))
        self.assertEqual("1i6g~", librime_check.keys_for(["ㄅㄛˊ", "ㄕ~"], zhuyin))
        self.assertIsNone(librime_check.keys_for(["invalid"], zhuyin))

    def test_strip_tone_handles_both_code_formats(self):
        self.assertEqual("ce", layer.strip_tone("ce4"))
        self.assertEqual("ㄦ", layer.strip_tone("ㄦ˙"))
        self.assertEqual("ㄕ~", layer.strip_tone("ㄕ~"))

    def test_toneless_layer_matches_zhuyin_readings(self):
        engine = layer.Layer(
            [("甲乙", "ㄘㄜˋ ㄕˋ", 3)],
            {"丙": {"ㄘㄜˇ": 100.0}, "丁": {"ㄕˊ": 100.0}},
            {},
            tone=False,
        )

        output, _fires = engine.apply("丙丁")

        self.assertEqual("甲乙", output)


if __name__ == "__main__":
    unittest.main()
