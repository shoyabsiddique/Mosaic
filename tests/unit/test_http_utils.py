import requests

from decision_engine.data.http_utils import get_with_retry


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")


def test_succeeds_on_first_try_without_sleeping(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))

    resp = get_with_retry("http://example.com")
    assert resp.status_code == 200
    assert len(calls) == 1


def test_retries_after_transient_failure_then_succeeds(monkeypatch):
    attempts = {"n": 0}
    sleeps = []

    def fake_get(url, params=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return _FakeResponse(502)
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    resp = get_with_retry("http://example.com", retries=2)
    assert resp.status_code == 200
    assert attempts["n"] == 3
    assert len(sleeps) == 2  # slept between attempts 1->2 and 2->3


def test_raises_last_error_after_exhausting_retries(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        return _FakeResponse(502)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: None)

    try:
        get_with_retry("http://example.com", retries=2)
        assert False, "expected HTTPError"
    except requests.exceptions.HTTPError:
        pass


def test_connection_error_is_also_retried(monkeypatch):
    attempts = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise requests.exceptions.ConnectionError("boom")
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: None)

    resp = get_with_retry("http://example.com", retries=2)
    assert resp.status_code == 200
    assert attempts["n"] == 2


def test_params_and_headers_are_forwarded(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["params"] = params
        seen["headers"] = headers
        seen["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    get_with_retry("http://example.com", params={"a": 1}, headers={"X": "y"}, timeout=45)
    assert seen == {"params": {"a": 1}, "headers": {"X": "y"}, "timeout": 45}
