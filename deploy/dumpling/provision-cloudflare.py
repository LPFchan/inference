#!/usr/bin/env python3
"""Provision dedicated STT ingress. Read the Cloudflare API token from stdin."""
import json
import argparse
from pathlib import Path
import sys
import urllib.error
import urllib.request

HOSTNAME = "transcribe.lost.plus"
TUNNEL_NAME = "dumpling-stt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tunnel-name", default=TUNNEL_NAME)
    args = parser.parse_args()
    tunnel_name = args.tunnel_name
    key = sys.stdin.readline().strip()
    if not key:
        raise SystemExit("A Cloudflare API token is required on stdin")

    def api(method, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            "https://api.cloudflare.com/client/v4" + path, data=data, method=method,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.load(response)
        if not result.get("success"):
            raise RuntimeError(str(result.get("errors")))
        return result["result"]

    zone = api("GET", "/zones?name=lost.plus")[0]
    account, zone_id = zone["account"]["id"], zone["id"]
    records = api("GET", f"/zones/{zone_id}/dns_records?name={HOSTNAME}")
    tunnels = api("GET", f"/accounts/{account}/cfd_tunnel?is_deleted=false&name={tunnel_name}")
    if tunnels:
        tunnel = tunnels[0]
    else:
        if records:
            raise RuntimeError("Hostname already has a DNS record; refusing to replace it")
        tunnel = api("POST", f"/accounts/{account}/cfd_tunnel",
                     {"name": tunnel_name, "config_src": "cloudflare"})
    tunnel_id = tunnel["id"]
    target = tunnel_id + ".cfargotunnel.com"
    if any(record["type"] != "CNAME" or record["content"] != target for record in records):
        raise RuntimeError("Conflicting DNS record")
    api("PUT", f"/accounts/{account}/cfd_tunnel/{tunnel_id}/configurations", {
        "config": {"ingress": [
            {"hostname": HOSTNAME, "service": "http://127.0.0.1:8792"},
            {"service": "http_status:404"},
        ]},
    })
    if not records:
        api("POST", f"/zones/{zone_id}/dns_records",
            {"type": "CNAME", "name": "transcribe", "content": target, "proxied": True, "ttl": 1})
    token = api("GET", f"/accounts/{account}/cfd_tunnel/{tunnel_id}/token")
    folder = Path.home() / ".config/inference"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "stt-tunnel-token"
    # Set permissions before writing a newly created token file.
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(token + "\n")
    rule = {
        "ref": tunnel_name.replace("-", "_") + "_api_bic",
        "description": "Allow ordinary HTTP clients for the authenticated STT API",
        "expression": 'http.host eq "transcribe.lost.plus"',
        "action": "set_config", "action_parameters": {"bic": False}, "enabled": True,
    }
    try:
        ruleset = api("GET", f"/zones/{zone_id}/rulesets/phases/http_config_settings/entrypoint")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        ruleset = api("POST", f"/zones/{zone_id}/rulesets", {
            "name": "Configuration rules", "kind": "zone", "phase": "http_config_settings", "rules": [rule],
        })
    existing = next((item for item in ruleset.get("rules", []) if item.get("ref") == rule["ref"]), None)
    if existing is None:
        api("POST", f'/zones/{zone_id}/rulesets/{ruleset["id"]}/rules', rule)
    elif any(existing.get(field) != rule[field] for field in ("expression", "action", "action_parameters", "enabled")):
        api("PATCH", f'/zones/{zone_id}/rulesets/{ruleset["id"]}/rules/{existing["id"]}', rule)
    print(json.dumps({"tunnel_id": tunnel_id, "hostname": HOSTNAME,
                      "configuration_ruleset_id": ruleset["id"]}), flush=True)


if __name__ == "__main__":
    main()
