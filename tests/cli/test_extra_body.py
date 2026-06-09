"""Tests for the --extra-body / -x CLI flag and its supporting machinery.

Covers:
  - _parse_extra_body validation and rejection
  - _resolve_turn_agent_config composition with service-tier overrides
  - ChatCompletionsTransport legacy path extra_body merge
  - argparse parser accepts -x/--extra-body
  - cmd_chat forwards extra_body into cli.main kwargs
  - TUI + --extra-body rejection
"""

import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# _parse_extra_body tests
# ============================================================================

class TestParseExtraBody(unittest.TestCase):
    """Test the _parse_extra_body() helper in cli.py."""

    def _import_cli(self):
        import hermes_cli.config as config_mod

        if not hasattr(config_mod, "save_env_value_secure"):
            config_mod.save_env_value_secure = lambda key, value: {
                "success": True,
                "stored_as": key,
                "validated": False,
            }

        import cli as cli_mod
        return cli_mod

    def test_none_returns_none(self):
        cli_mod = self._import_cli()
        self.assertIsNone(cli_mod._parse_extra_body(None))

    def test_empty_string_returns_none(self):
        cli_mod = self._import_cli()
        self.assertIsNone(cli_mod._parse_extra_body(""))

    def test_pre_parsed_dict_passes_through(self):
        cli_mod = self._import_cli()
        d = {"provider": {"only": ["anthropic"]}}
        self.assertEqual(cli_mod._parse_extra_body(d), d)

    def test_valid_json_object(self):
        cli_mod = self._import_cli()
        result = cli_mod._parse_extra_body('{"provider":{"only":["anthropic"]}}')
        self.assertEqual(result, {"provider": {"only": ["anthropic"]}})

    def test_invalid_json_raises_valueerror(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body("{bad json}")
        self.assertIn("valid JSON", str(ctx.exception))

    def test_array_rejected(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body("[]")
        self.assertIn("decode to a JSON object", str(ctx.exception))

    def test_string_rejected(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body('"foo"')
        self.assertIn("decode to a JSON object", str(ctx.exception))

    def test_number_rejected(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body("123")
        self.assertIn("decode to a JSON object", str(ctx.exception))

    def test_bool_rejected(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body("true")
        self.assertIn("decode to a JSON object", str(ctx.exception))

    def test_null_rejected(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body("null")
        self.assertIn("decode to a JSON object", str(ctx.exception))

    def test_non_string_raises_valueerror(self):
        cli_mod = self._import_cli()
        with self.assertRaises(ValueError) as ctx:
            cli_mod._parse_extra_body(42)
        self.assertIn("must be a JSON object", str(ctx.exception))


# ============================================================================
# _resolve_turn_agent_config tests
# ============================================================================

class TestResolveTurnAgentConfig(unittest.TestCase):
    """Test that extra_body composes correctly with service-tier overrides."""

    def _make_self(self, *, service_tier=None, extra_body=None):
        """Build a mock HermesCLI-like object with the needed attributes."""
        return SimpleNamespace(
            api_key="test-key",
            base_url="https://api.openai.com",
            provider="openrouter",
            api_mode="chat_completions",
            acp_command=None,
            acp_args=[],
            model="test-model",
            service_tier=service_tier,
            extra_body=extra_body,
        )

    def _resolve(self, self_obj):
        """Import and call the real _resolve_turn_agent_config."""
        import hermes_cli.cli_agent_setup_mixin as mixin

        # Bind the method to our mock self
        return mixin.CLIAgentSetupMixin._resolve_turn_agent_config(
            self_obj, user_message="hello"
        )

    def test_no_service_tier_no_extra_body(self):
        self_obj = self._make_self(service_tier=None, extra_body=None)
        result = self._resolve(self_obj)
        self.assertIsNone(result["request_overrides"])

    def test_no_service_tier_with_extra_body(self):
        self_obj = self._make_self(
            service_tier=None,
            extra_body={"provider": {"only": ["anthropic"]}},
        )
        result = self._resolve(self_obj)
        self.assertEqual(
            result["request_overrides"],
            {"extra_body": {"provider": {"only": ["anthropic"]}}},
        )

    def test_service_tier_with_extra_body(self):
        self_obj = self._make_self(
            service_tier="priority",
            extra_body={"provider": {"only": ["anthropic"]}},
        )
        result = self._resolve(self_obj)
        # Fast-mode overrides are model-dependent and resolve_fast_mode_overrides
        # may return None for unknown models, so we at least verify extra_body
        # is present even when service_tier is active.
        overrides = result["request_overrides"]
        self.assertIsNotNone(overrides)
        self.assertEqual(
            overrides["extra_body"],
            {"provider": {"only": ["anthropic"]}},
        )

    def test_service_tier_extra_body_present_even_when_fast_overrides_none(self):
        """When resolve_fast_mode_overrides returns None, extra_body must survive."""
        self_obj = self._make_self(
            service_tier="priority",
            extra_body={"tags": ["test-tag"]},
        )
        with patch(
            "hermes_cli.models.resolve_fast_mode_overrides",
            return_value=None,
        ):
            result = self._resolve(self_obj)
        self.assertEqual(
            result["request_overrides"],
            {"extra_body": {"tags": ["test-tag"]}},
        )

    def test_service_tier_active_fast_overrides_and_extra_body_composed(self):
        """Both fast overrides and extra_body should be present."""
        self_obj = self._make_self(
            service_tier="priority",
            extra_body={"tags": ["test-tag"]},
        )
        fast = {"service_tier": "priority"}
        with patch(
            "hermes_cli.models.resolve_fast_mode_overrides",
            return_value=fast,
        ):
            result = self._resolve(self_obj)
        self.assertEqual(
            result["request_overrides"],
            {"service_tier": "priority", "extra_body": {"tags": ["test-tag"]}},
        )


# ============================================================================
# ChatCompletionsTransport legacy path extra_body merge tests
# ============================================================================

class TestLegacyExtraBodyMerge:
    """Test that the legacy transport path merges extra_body correctly."""

    @pytest.fixture
    def transport(self):
        from agent.transports.chat_completions import ChatCompletionsTransport
        return ChatCompletionsTransport()

    _messages = [{"role": "user", "content": "hello"}]

    def _simple_params(self, **overrides):
        return {
            "reasoning_config": overrides.pop("reasoning_config", None),
            "request_overrides": overrides.pop("request_overrides", None),
        }

    def test_existing_extra_body_keys_preserved(self, transport):
        """Existing non-conflicting extra_body keys survive the override."""
        params = self._simple_params(
            reasoning_config={"effort": "medium"},
            request_overrides={"extra_body": {"provider": {"only": ["anthropic"]}}},
        )
        kwargs = transport.build_kwargs(
            "test-model", self._messages, **params
        )
        extra = kwargs.get("extra_body")
        assert extra is not None
        assert "provider" in extra
        assert extra["provider"] == {"only": ["anthropic"]}

    def test_user_keys_win_on_conflict(self, transport):
        """User-supplied extra_body keys override existing ones at the top level."""
        params = self._simple_params(
            request_overrides={
                "extra_body": {"provider": {"only": ["anthropic"]}}
            },
        )
        kwargs = transport.build_kwargs(
            "test-model", self._messages, **params
        )
        assert kwargs["extra_body"]["provider"] == {"only": ["anthropic"]}

    def test_non_extra_body_overrides_still_apply(self, transport):
        """Non-extra_body override keys are applied normally."""
        params = self._simple_params(
            request_overrides={"service_tier": "priority"},
        )
        kwargs = transport.build_kwargs(
            "test-model", self._messages, **params
        )
        assert kwargs.get("service_tier") == "priority"

    def test_request_overrides_none_does_nothing(self, transport):
        """A None request_overrides should not break build_kwargs."""
        params = self._simple_params(request_overrides=None)
        kwargs = transport.build_kwargs(
            "test-model", self._messages, **params
        )
        assert "extra_body" not in kwargs  # no reasoning → no extra_body


# ============================================================================
# Parser acceptance tests
# ============================================================================

class TestParserExtraBodyArg:
    """Test that argparse accepts -x/--extra-body on the chat subparser."""

    @pytest.fixture
    def parser(self):
        from hermes_cli._parser import build_top_level_parser
        _, _, chat_parser = build_top_level_parser()
        return chat_parser

    def test_long_flag_accepted(self, parser):
        args = parser.parse_args(["--extra-body", '{"provider":{"only":["a"]}}'])
        assert args.extra_body == '{"provider":{"only":["a"]}}'

    def test_short_flag_accepted(self, parser):
        args = parser.parse_args(["-x", '{"provider":{"only":["a"]}}'])
        assert args.extra_body == '{"provider":{"only":["a"]}}'

    def test_default_is_none(self, parser):
        args = parser.parse_args([])
        assert args.extra_body is None


# ============================================================================
# cmd_chat kwargs forwarding test
# ============================================================================

class TestCmdChatForwarding(unittest.TestCase):
    """Test that cmd_chat forwards extra_body into cli.main() kwargs."""

    def test_extra_body_forwarded(self):
        """cmd_chat(args) should include extra_body in the kwargs dict."""
        from hermes_cli.main import cmd_chat

        # Build a mock args that forces the classic CLI path
        args = SimpleNamespace(
            model=None,
            provider=None,
            toolsets=None,
            skills=None,
            verbose=None,
            quiet=False,
            query="hello",
            image=None,
            resume=None,
            worktree=False,
            checkpoints=False,
            pass_session_id=False,
            max_turns=None,
            ignore_rules=False,
            compact=False,
            extra_body='{"provider":{"only":["anthropic"]}}',
            source=None,
            tui=False,
            tui_dev=False,
            cli=False,
            yolo=False,
            accept_hooks=False,
            ignore_user_config=False,
            continue_last=None,
        )

        # cli_main is imported as `from cli import main as cli_main` inside cmd_chat,
        # so we patch cli.main.
        with patch("cli.main") as mock_cli_main:
            with patch("hermes_cli.main._has_any_provider_configured", return_value=True):
                cmd_chat(args)

        mock_cli_main.assert_called_once()
        call_kwargs = mock_cli_main.call_args[1]
        assert call_kwargs.get("extra_body") == '{"provider":{"only":["anthropic"]}}'

    def test_tui_rejects_extra_body(self):
        """TUI path with --extra-body should print error and exit."""
        from hermes_cli.main import cmd_chat

        args = SimpleNamespace(
            model=None,
            provider=None,
            toolsets=None,
            skills=None,
            verbose=None,
            quiet=False,
            query="hello",
            image=None,
            resume=None,
            worktree=False,
            checkpoints=False,
            pass_session_id=False,
            max_turns=None,
            ignore_rules=False,
            compact=False,
            extra_body='{"provider":{"only":["anthropic"]}}',
            source=None,
            tui=True,  # <-- TUI requested
            tui_dev=False,
            cli=False,
            yolo=False,
            accept_hooks=False,
            ignore_user_config=False,
            continue_last=None,
        )

        with patch("hermes_cli.main._has_any_provider_configured", return_value=True):
            with self.assertRaises(SystemExit) as ctx:
                cmd_chat(args)
            self.assertEqual(ctx.exception.code, 1)


# ============================================================================
# cli.main() accepts extra_body and passes it to HermesCLI
# ============================================================================

class TestCliMainExtraBody(unittest.TestCase):
    """Test that cli.main() passes extra_body to HermesCLI."""

    def test_extra_body_passed_to_hermescli(self):
        """cli.main() should pass extra_body= into HermesCLI(...)."""
        import hermes_cli.config as config_mod

        if not hasattr(config_mod, "save_env_value_secure"):
            config_mod.save_env_value_secure = lambda key, value: {
                "success": True,
                "stored_as": key,
                "validated": False,
            }

        import cli as cli_mod

        with patch.object(cli_mod, "HermesCLI") as mock_cli:
            mock_instance = MagicMock()
            mock_cli.return_value = mock_instance

            # Force an early return so we don't call run()
            with patch.object(cli_mod, "_git_repo_root", return_value=None):
                with patch("sys.exit") as mock_exit:
                    mock_exit.side_effect = SystemExit(0)
                    try:
                        cli_mod.main(
                            query="hello",
                            extra_body='{"provider":{"only":["a"]}}',
                        )
                    except SystemExit:
                        pass

        mock_cli.assert_called_once()
        call_kwargs = mock_cli.call_args[1]
        assert call_kwargs.get("extra_body") == '{"provider":{"only":["a"]}}'
