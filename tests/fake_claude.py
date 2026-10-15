#!/usr/bin/env python3
"""Stand-in for the `claude` CLI, used by the tests through CLAUDE_CLI_PATH.

Behaviour is chosen with FAKE_CLI_MODE. Every call appends a JSON record (argv, cwd, whether the
API key is visible, stdin) to FAKE_CLI_LOG when that variable is set.
"""

import json
import os
import sys
import time

ARGS = sys.argv[1:]
MODE = os.environ.get("FAKE_CLI_MODE", "ok")


def emit(event):
    if os.environ.get("FAKE_CLI_MODE") == "no_cost" and event.get("type") == "result":
        event = {k: v for k, v in event.items() if k != "total_cost_usd"}
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def result(**overrides):
    base = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "PONG",
        "structured_output": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_read_input_tokens": 50,
            "cache_creation_input_tokens": 10,
            "server_tool_use": {"web_search_requests": 0},
        },
        "total_cost_usd": 0.0012,
        "modelUsage": {"claude-haiku-5-5": {"inputTokens": 100}},
        "permission_denials": [],
        "stop_reason": "end_turn",
    }
    base.update(overrides)
    return base


def main():
    mode = MODE
    if "--version" in ARGS:
        print("2.1.0 (Claude Code)")
        return 0
    if ARGS[:2] == ["auth", "status"]:
        print(json.dumps({"loggedIn": mode != "not_logged_in", "subscriptionType": "pro"}))
        return 0

    stdin_text = sys.stdin.read()
    blocks = []
    for line in stdin_text.splitlines():
        if line.strip():
            blocks = json.loads(line)["message"]["content"]
    log = os.environ.get("FAKE_CLI_LOG")
    if log:
        with open(log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({
                "args": ARGS,
                "cwd": os.getcwd(),
                "env_has_api_key": "ANTHROPIC_API_KEY" in os.environ,
                "env_has_auth_token": "ANTHROPIC_AUTH_TOKEN" in os.environ,
                "env_has_base_url": "ANTHROPIC_BASE_URL" in os.environ,
                "blocks": blocks,
                "mode": mode,
            }) + "\n")

    if mode == "rate_limit_then_ok":
        counter_path = os.environ["FAKE_RATE_COUNTER"]
        count = int(open(counter_path).read()) if os.path.exists(counter_path) else 0
        with open(counter_path, "w") as handle:
            handle.write(str(count + 1))
        if count < int(os.environ.get("FAKE_RATE_FAILURES", "2")):
            emit(result(is_error=True, result="Rate limit exceeded", api_error_status=429, subtype="error"))
            return 1
        mode = "ok"

    wants_schema = "--json-schema" in ARGS
    web = "WebSearch" in ARGS[ARGS.index("--tools") + 1] if "--tools" in ARGS else False
    emit({"type": "system", "subtype": "init"})

    if mode == "garbage":
        print("this is not json")
        return 0
    if mode == "timeout":
        time.sleep(30)
        return 0
    if mode == "timeout_with_child":
        import subprocess

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        with open(os.environ["FAKE_CHILD_PIDFILE"], "w") as handle:
            handle.write(str(child.pid))
        time.sleep(60)
        return 0
    if mode == "stderr_error":
        sys.stderr.write(os.environ.get("FAKE_CLI_STDERR", "error") + "\n")
        return 1
    if mode == "sleep":
        time.sleep(float(os.environ.get("FAKE_CLI_SLEEP", "0.5")))
    if mode == "is_error":
        emit(result(is_error=True, result="Something broke", subtype="error_during_execution"))
        return 1
    if mode == "usage_limit":
        emit(result(is_error=True, result="You have reached your usage limit. Try again later.", subtype="error"))
        return 1
    if mode == "rate_limit":
        emit(result(is_error=True, result="Rate limit exceeded", api_error_status=429, subtype="error"))
        return 1
    if mode == "not_logged_in":
        emit(result(is_error=True, result="Not logged in · Please run /login", subtype="error"))
        return 1
    if mode == "no_models":
        emit(result(result="", modelUsage={}))
        return 0

    if web:
        emit({"type": "assistant", "message": {"content": [{"type": "tool_use", "name": "WebSearch"}]}})
        links = json.dumps([{"title": "Spec page", "url": "https://specs.example.com/acme-x1"}])
        emit({"type": "user", "message": {"content": [{
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": f'Web search results for query: "acme x1"\n\nLinks: {links}',
        }]}})
        if mode == "websearch_denied":
            emit(result(result="From memory: 24 MP.", permission_denials=[{"tool_name": "WebSearch"}]))
            return 0
        emit(result(result="Resolution 24 MP per the spec page. Sensor APS-C."))
        return 0

    if wants_schema:
        if mode == "invalid_then_ok":
            counter_path = os.environ["FAKE_CLI_COUNTER"]
            count = int(open(counter_path).read()) if os.path.exists(counter_path) else 0
            with open(counter_path, "w") as handle:
                handle.write(str(count + 1))
            answer = {"ok": "maybe"} if count == 0 else {"ok": True}
        else:
            answer = json.loads(os.environ.get("FAKE_CLI_ANSWER", '{"ok": true}'))
        emit(result(result=json.dumps(answer), structured_output=answer))
        return 0

    emit(result(result="PONG"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
