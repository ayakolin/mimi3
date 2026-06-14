import json
import unittest

from mimo2api import web_service

MARKER = web_service.OPENCLAW_SYSTEM_MARKER


class OpenClawV25ProCompatTests(unittest.TestCase):
    def test_injects_system_marker_when_no_system(self):
        body = json.dumps({"model": "mimo-v2.5-pro", "messages": [{"role": "user", "content": "hi"}]})

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))

        self.assertEqual(out["messages"][0], {"role": "system", "content": MARKER})
        self.assertEqual(out["messages"][1], {"role": "user", "content": "hi"})

    def test_appends_marker_to_existing_system(self):
        body = json.dumps({"model": "mimo-v2.5-pro", "messages": [
            {"role": "system", "content": "You are a coding assistant."},
            {"role": "user", "content": "hi"},
        ]})

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))
        sys_content = out["messages"][0]["content"]

        self.assertIn("You are a coding assistant.", sys_content)
        self.assertIn(MARKER, sys_content)

    def test_idempotent_when_marker_already_present(self):
        existing = f"Custom rules.\n\n{MARKER}"
        body = json.dumps({"model": "mimo-v2.5-pro", "messages": [
            {"role": "system", "content": existing},
            {"role": "user", "content": "hi"},
        ]})

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))

        self.assertEqual(out["messages"][0]["content"].count(MARKER), 1)

    def test_covers_prefixed_model_id(self):
        body = json.dumps({"model": "xiaomi/mimo-v2.5-pro", "messages": [{"role": "user", "content": "hi"}]})

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))

        self.assertEqual(out["messages"][0]["role"], "system")
        self.assertIn(MARKER, out["messages"][0]["content"])

    def test_preserves_tools_field(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        body = json.dumps({"model": "mimo-v2.5-pro", "messages": [{"role": "user", "content": "hi"}], "tools": tools})

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))

        self.assertEqual(out["tools"], tools)

    def test_injects_into_anthropic_top_level_system(self):
        body = json.dumps({"model": "mimo-v2.5-pro", "system": "Be terse.", "messages": []})

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))

        self.assertIn("Be terse.", out["system"])
        self.assertIn(MARKER, out["system"])

    def test_keeps_other_models_unchanged(self):
        original = {"model": "mimo-v2.5", "messages": [{"role": "user", "content": "hi"}]}
        body = json.dumps(original)

        out = json.loads(web_service.apply_openclaw_v25pro_compat(body))

        self.assertEqual(out, original)

    def test_keeps_invalid_json_unchanged(self):
        body = "{not-json"

        self.assertEqual(web_service.apply_openclaw_v25pro_compat(body), body)


if __name__ == "__main__":
    unittest.main()
