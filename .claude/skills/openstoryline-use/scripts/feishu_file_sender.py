#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests


FEISHU_TOKEN_URL = (
    "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
)
FEISHU_UPLOAD_URL = "https://open.feishu.cn/open-apis/im/v1/files"
FEISHU_SEND_MSG_URL = "https://open.feishu.cn/open-apis/im/v1/messages"
OPENCLAW_CONFIG = Path.home() / ".openclaw" / "openclaw.json"


def load_openclaw_config() -> Dict[str, Any]:
    if not OPENCLAW_CONFIG.exists():
        raise FileNotFoundError(f"OpenClaw config not found: {OPENCLAW_CONFIG}")
    return json.loads(OPENCLAW_CONFIG.read_text(encoding="utf-8"))


def is_path_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_agent_id(config: Dict[str, Any]) -> str:
    cwd = Path.cwd().resolve()
    best_match = (0, None)
    for agent in config.get("agents", {}).get("list", []):
        workspace = agent.get("workspace")
        agent_id = agent.get("id")
        if not workspace or not agent_id:
            continue
        workspace_path = Path(workspace).resolve()
        if is_path_within(cwd, workspace_path):
            match_len = len(str(workspace_path))
            if match_len > best_match[0]:
                best_match = (match_len, agent_id)
    if best_match[1]:
        return best_match[1]

    default_agent = (
        config.get("agent", {}).get("default")
        or config.get("agents", {}).get("default")
        or config.get("agents", {}).get("defaults")
    )
    if isinstance(default_agent, dict):
        workspace = default_agent.get("workspace")
        agent_id = default_agent.get("id")
        if workspace and agent_id:
            workspace_path = Path(workspace).resolve()
            if is_path_within(cwd, workspace_path):
                return agent_id

    raise RuntimeError("Unable to resolve agent id from workspace path")


def resolve_feishu_account(
    config: Dict[str, Any], agent_id: Optional[str] = None
) -> Tuple[str, str]:
    feishu_config = config.get("channels", {}).get("feishu", {})
    app_id = feishu_config.get("appId")
    app_secret = feishu_config.get("appSecret")
    if app_id and app_secret:
        return app_id, app_secret

    if agent_id is None:
        agent_id = resolve_agent_id(config)

    bindings = config.get("bindings", [])
    account_id = None
    for binding in bindings:
        if binding.get("agentId") == agent_id:
            account_id = binding.get("match", {}).get("accountId")
            if account_id:
                break
    if not account_id:
        raise RuntimeError(f"No Feishu account binding for agent: {agent_id}")

    accounts = (
        feishu_config
        .get("accounts", {})
    )
    account = accounts.get(account_id)
    if not account:
        raise RuntimeError(f"Feishu account not found: {account_id}")
    account_app_id = account.get("appId")
    account_app_secret = account.get("appSecret")
    if not account_app_id or not account_app_secret:
        raise RuntimeError(f"Missing appId/appSecret for account: {account_id}")
    return account_app_id, account_app_secret


def get_tenant_access_token(app_id: str, app_secret: str) -> str:
    resp = requests.post(
        FEISHU_TOKEN_URL,
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Get token failed: {data}")
    return data["tenant_access_token"]


def upload_file(token: str, file_path: Path, file_type: str) -> str:
    headers = {"Authorization": f"Bearer {token}"}
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f)}
        data = {
            "file_type": file_type,
            "file_name": file_path.name,
        }
        resp = requests.post(
            FEISHU_UPLOAD_URL,
            headers=headers,
            data=data,
            files=files,
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Upload failed: {data}")
    return data["data"]["file_key"]


def send_file_message(
    token: str,
    receive_id: str,
    receive_id_type: str,
    file_key: str,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json; charset=utf-8",
    }
    params = {"receive_id_type": receive_id_type}
    payload = {
        "receive_id": receive_id,
        "msg_type": "file",
        "content": json.dumps({"file_key": file_key}),
    }
    resp = requests.post(
        FEISHU_SEND_MSG_URL,
        headers=headers,
        params=params,
        json=payload,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Send message failed: {data}")
    return data


def infer_receive_id_type(receive_id: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    if receive_id.startswith("oc_"):
        return "chat_id"
    if receive_id.startswith("ou_"):
        return "open_id"
    if receive_id.startswith("on_"):
        return "user_id"
    return "chat_id"


def resolve_receive_id(cli_value: Optional[str]) -> str:
    if cli_value:
        return cli_value
    env_value = (
        os.getenv("OPENCLAW_CHAT_ID")
        or os.getenv("OPENCLAW_RECEIVE_ID")
        or os.getenv("FEISHU_CHAT_ID")
    )
    if env_value:
        return env_value
    raise RuntimeError(
        "Missing receive_id. Provide --receive-id or set OPENCLAW_CHAT_ID."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload file to Feishu and send")
    parser.add_argument("--file", required=True, help="Local file path")
    parser.add_argument("--receive-id", default=None, help="chat_id or open_id")
    parser.add_argument(
        "--receive-id-type",
        default=None,
        help="chat_id / open_id / user_id (auto-detect if omitted)",
    )
    parser.add_argument(
        "--file-type",
        default="stream",
        help="file_type for upload, default stream",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    file_path = Path(args.file)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    config = load_openclaw_config()
    app_id, app_secret = resolve_feishu_account(config)

    receive_id = resolve_receive_id(args.receive_id)
    receive_id_type = infer_receive_id_type(receive_id, args.receive_id_type)

    token = get_tenant_access_token(app_id, app_secret)
    file_key = upload_file(token, file_path, args.file_type)
    result = send_file_message(token, receive_id, receive_id_type, file_key)
    print("Send success:", result)


if __name__ == "__main__":
    main()
