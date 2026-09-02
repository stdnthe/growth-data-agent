from __future__ import annotations

import unittest
from unittest.mock import patch

import app
from agent.intent_router import IntentRoutingError


class AppRuntimeTests(unittest.TestCase):
    def test_public_deepseek_config_uses_official_runtime(self) -> None:
        self.assertEqual(app.PUBLIC_LLM_DEFAULTS["deepseek"]["model"], "deepseek-chat")
        self.assertEqual(
            app.PUBLIC_LLM_DEFAULTS["deepseek"]["base_url"],
            "https://api.deepseek.com/v1",
        )

    def test_analysis_error_survives_the_submit_rerun(self) -> None:
        session_state: dict[str, object] = {
            "last_analysis_run": object(),
            "pending_clarification": object(),
        }
        with (
            patch.object(app.st, "session_state", session_state),
            patch.object(
                app,
                "execute_analysis",
                side_effect=IntentRoutingError("invalid API key"),
            ),
        ):
            app._start_analysis("近30天GMV趋势", app.PipelineConfig("db", "metrics"))

        self.assertEqual(session_state["analysis_error"], "invalid API key")
        self.assertNotIn("last_analysis_run", session_state)
        self.assertNotIn("pending_clarification", session_state)


if __name__ == "__main__":
    unittest.main()
