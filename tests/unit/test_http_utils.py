import requests

from decision_engine.data.http_utils import DEFAULT_USER_AGENT, get_with_retry


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            err = requests.exceptions.HTTPError(f"{self.status_code} error")
            err.response = self  # real requests.HTTPError carries this; needed for status-based logic
            raise err


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


def test_retries_after_transient_5xx_then_succeeds(monkeypatch):
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


def test_raises_last_error_after_exhausting_retries_on_5xx(monkeypatch):
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


def test_403_fails_immediately_without_retrying(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse(403)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))

    try:
        get_with_retry("http://example.com", retries=2)
        assert False, "expected HTTPError"
    except requests.exceptions.HTTPError:
        pass
    assert len(calls) == 1  # no retries attempted at all


def test_404_also_fails_immediately_without_retrying(monkeypatch):
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        return _FakeResponse(404)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: (_ for _ in ()).throw(AssertionError("should not sleep")))

    try:
        get_with_retry("http://example.com", retries=2)
        assert False, "expected HTTPError"
    except requests.exceptions.HTTPError:
        pass
    assert len(calls) == 1


def test_429_is_retried_like_5xx(monkeypatch):
    attempts = {"n": 0}

    def fake_get(url, params=None, headers=None, timeout=None):
        attempts["n"] += 1
        if attempts["n"] < 2:
            return _FakeResponse(429)
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda s: None)

    resp = get_with_retry("http://example.com", retries=2)
    assert resp.status_code == 200
    assert attempts["n"] == 2


def test_default_user_agent_is_sent_when_no_headers_given(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["headers"] = headers
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    get_with_retry("http://example.com")
    assert seen["headers"]["User-Agent"] == DEFAULT_USER_AGENT


def test_caller_headers_are_merged_with_default_user_agent(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["headers"] = headers
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    get_with_retry("http://example.com", headers={"Accept": "application/json"})
    assert seen["headers"]["User-Agent"] == DEFAULT_USER_AGENT
    assert seen["headers"]["Accept"] == "application/json"


def test_caller_can_override_the_default_user_agent(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["headers"] = headers
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    get_with_retry("http://example.com", headers={"User-Agent": "custom-agent"})
    assert seen["headers"]["User-Agent"] == "custom-agent"


def test_params_and_timeout_are_forwarded(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["params"] = params
        seen["timeout"] = timeout
        return _FakeResponse(200)

    monkeypatch.setattr(requests, "get", fake_get)
    get_with_retry("http://example.com", params={"a": 1}, timeout=45)
    assert seen["params"] == {"a": 1}
    assert seen["timeout"] == 45
