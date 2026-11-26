"""Cache-Control headers for the served Chat web SPA."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from grimoire.entrypoint import _WebUIStaticFiles


def test_webui_html_revalidates_and_hashed_assets_are_immutable(tmp_path):
    (tmp_path / "index.html").write_text("<!doctype html><title>Chat</title>", encoding="utf-8")
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "index-AbCd1234.js").write_text("console.log(1);", encoding="utf-8")
    (tmp_path / "robots.txt").write_text("User-agent: *", encoding="utf-8")

    app = FastAPI()
    app.mount("/", _WebUIStaticFiles(directory=str(tmp_path), html=True), name="webui")

    with TestClient(app) as client:
        root = client.get("/")
        assert root.status_code == 200
        assert root.headers["cache-control"] == "no-cache"
        assert "etag" in root.headers

        root_304 = client.get("/", headers={"If-None-Match": root.headers["etag"]})
        assert root_304.status_code == 304
        assert root_304.headers["cache-control"] == "no-cache"

        index_resp = client.get("/index.html")
        assert index_resp.status_code == 200
        assert index_resp.headers["cache-control"] == "no-cache"

        head_root = client.head("/")
        assert head_root.status_code == 200
        assert head_root.headers["cache-control"] == "no-cache"

        asset = client.get("/assets/index-AbCd1234.js")
        assert asset.status_code == 200
        assert asset.headers["cache-control"] == "private, max-age=31536000, immutable"

        asset_304 = client.get(
            "/assets/index-AbCd1234.js",
            headers={"If-None-Match": asset.headers["etag"]},
        )
        assert asset_304.status_code == 304
        assert asset_304.headers["cache-control"] == "private, max-age=31536000, immutable"

        other = client.get("/robots.txt")
        assert other.status_code == 200
        assert "cache-control" not in other.headers
