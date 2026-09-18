"""Unit tests for config-aware reconstruction user prompts.

These cover the single-shot path only. Graph context reaches the model through
the agentic path (``agent_predictor``), never through ``build_user_prompt``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.experiment import _ocel_text_for, _strip_ocel_generator_header
from src.prompt_store import P
from src.sequence_predictor import build_user_prompt

CATALOG = "- A_Accepted: application accepted\n- O_Created: offer record exists"

GENERATOR_TIMELINE = """\
OBSERVED CASE TIMELINE (offline): every '-' line below is OBSERVED. \
Events BEFORE and AFTER the hidden email are given - use BOTH sides. \
Reconstruct ONLY the email marked THIS.

- 2016-03-10 | A_Submitted | objects: app1
>>> THIS hidden email TO RECONSTRUCT (activities unknown) <<<
- 2016-03-18 | O_Accepted | objects: app1
"""

RECORD = {"observed_timeline": GENERATOR_TIMELINE, "email_uid": "e1"}

MAIL = dict(
    subject="Update",
    body="I accepted the application.",
    sender="ada",
    objects="app1",
)


def _user(*, mail=False, ocel="") -> str:
    kw = dict(MAIL) if mail else dict(subject="", body="", sender="", objects="")
    return build_user_prompt(catalog=CATALOG, ocel_text=ocel, **kw)


def _assert_no_authority(text: str) -> None:
    assert "TRUST THE EMAIL" not in text
    assert "authoritative" not in text.lower()
    assert "email wins" not in text.lower()
    assert "use BOTH sides" not in text


def test_strip_generator_header() -> None:
    stripped = _strip_ocel_generator_header(GENERATOR_TIMELINE)
    assert "use BOTH sides" not in stripped
    assert "OBSERVED CASE TIMELINE" not in stripped
    assert ">>> THIS hidden email" in stripped
    assert "- 2016-03-10 | A_Submitted" in stripped
    ocel = _ocel_text_for(RECORD, use_ocel=True)
    assert "use BOTH sides" not in ocel
    assert ">>> THIS hidden email" in ocel


def test_ocel_withheld_when_disabled() -> None:
    assert _ocel_text_for(RECORD, use_ocel=False) == ""


def test_missing_timeline_raises() -> None:
    try:
        _ocel_text_for({"email_uid": "e1"}, use_ocel=True)
    except ValueError as exc:
        assert "observed_timeline" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a missing timeline")


def test_mail_only_email_plus_ocel() -> None:
    user = _user(mail=True, ocel=_ocel_text_for(RECORD, use_ocel=True))
    _assert_no_authority(user)
    assert "=== EMAIL ===" in user
    assert "I accepted the application." in user
    assert "=== OCEL CONTEXT" in user
    assert "do not copy them" in user
    _assert_no_authority(P("system"))


def test_ocel_only() -> None:
    user = _user(mail=False, ocel=_ocel_text_for(RECORD, use_ocel=True))
    _assert_no_authority(user)
    assert "=== EMAIL ===" not in user
    assert "=== OCEL CONTEXT" in user
    assert "do not copy them" in user
    assert ">>> THIS hidden email" in user


def test_mail_only_no_ocel() -> None:
    user = _user(mail=True, ocel="")
    _assert_no_authority(user)
    assert "=== EMAIL ===" in user
    assert "=== OCEL CONTEXT" not in user
