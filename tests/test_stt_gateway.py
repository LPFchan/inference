import asyncio

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient

from grimoire import config, stt_gateway
from grimoire.auth import AuthIdentity


def test_authenticated_upload_strips_credentials_and_private_headers(monkeypatch):
    seen = {}
    async def authenticate(request):
        seen['auth'] = request.headers.get('authorization')
        return AuthIdentity('1', '', '', 'user', ())
    async def backend(request):
        seen['headers'] = request.headers
        seen['body'] = await request.aread()
        return httpx.Response(200, json={'text': 'hello'}, headers={'x-stt-model': 'eg2', 'x-private': 'no'})
    original_client = httpx.AsyncClient
    monkeypatch.setattr(stt_gateway, 'authenticate_public', authenticate)
    monkeypatch.setattr(stt_gateway.httpx, 'AsyncClient', lambda **kwargs: original_client(transport=httpx.MockTransport(backend)))
    with TestClient(stt_gateway.app) as browser:
        result = browser.post('/v1/audio/transcriptions', data={'model': 'embeddinggemma-2-original'},
                              files={'file': ('a.wav', b'audio')},
                              headers={'authorization': 'Bearer secret', 'cookie': 'lp_auth=secret',
                                       'x-forwarded-for': 'spoof', 'x-lost-plus-sub': 'spoof'})
        assert result.status_code == 200, result.text
        assert result.json() == {'text': 'hello'}
        assert seen['auth'] == 'Bearer secret'
        assert result.headers['x-stt-model'] == 'eg2'
        assert 'x-private' not in result.headers
        assert b'audio' in seen['body']
        assert not any(header in seen['headers'] for header in ('authorization', 'cookie', 'x-forwarded-for', 'x-lost-plus-sub'))
        assert browser.post('/models/embeddinggemma-2-original/unload').status_code == 404


def test_failed_authentication_cannot_reach_backend(monkeypatch):
    async def reject(request):
        raise HTTPException(401, 'Authentication required')
    monkeypatch.setattr(stt_gateway, 'authenticate_public', reject)
    with TestClient(stt_gateway.app) as browser:
        assert browser.get('/v1/models').status_code == 401
        assert browser.post('/v1/audio/transcriptions', content=b'audio').status_code == 401
        assert browser.get('/health').status_code == 200


def test_cookie_upload_requires_exact_origin(monkeypatch):
    async def authenticate(request):
        return AuthIdentity('1', '', '', 'user', ())
    monkeypatch.setattr(stt_gateway, 'authenticate_public', authenticate)
    with TestClient(stt_gateway.app) as browser:
        assert browser.post('/v1/audio/transcriptions', headers={'origin': 'https://other.test'}).status_code == 403
        assert browser.post('/v1/audio/transcriptions').status_code == 403
