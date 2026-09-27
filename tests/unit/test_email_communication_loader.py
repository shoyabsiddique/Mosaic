from decision_engine.data.loaders.email_communication import (
    DOMAIN,
    _extract_subject_and_body,
    to_records_sms,
    to_records_spamassassin,
)

SMS_FIXTURE = [
    ("ham", "Hey, are we still on for lunch?"),
    ("spam", "WINNER! Claim your free prize now, call 555-0100"),
]


def test_sms_each_message_produces_exactly_one_record():
    records = list(to_records_sms(SMS_FIXTURE))
    assert len(records) == 2


def test_sms_ham_message_targets_not_spam():
    r = list(to_records_sms([SMS_FIXTURE[0]]))[0]
    assert r.target.label_index == 0
    assert r.target.distribution == (1.0, 0.0)


def test_sms_spam_message_targets_spam():
    r = list(to_records_sms([SMS_FIXTURE[1]]))[0]
    assert r.target.label_index == 1
    assert r.target.distribution == (0.0, 1.0)


def test_sms_state_is_the_raw_message_text():
    r = list(to_records_sms([SMS_FIXTURE[0]]))[0]
    assert r.state == SMS_FIXTURE[0][1]


def test_sms_domain_tagged_correctly():
    for r in to_records_sms(SMS_FIXTURE):
        assert r.domain == DOMAIN


SPAMASSASSIN_FIXTURE = [
    ("spam", "Buy now!!!", "Click here for free money, limited time offer."),
    ("easy_ham", "Meeting notes", "Here are the notes from today's standup."),
    ("hard_ham", "50% off newsletter unsubscribe info", "Thanks for subscribing, here's this week's roundup."),
]


def test_spamassassin_folder_maps_to_correct_category():
    records = {r.question_id: r for r in to_records_spamassassin([SPAMASSASSIN_FIXTURE[0]])}
    r = records["spamassassin-0-category"]
    labels = r.question.option_labels()
    assert r.target.label_index == labels.index("spam")


def test_spamassassin_easy_ham_maps_to_legitimate_not_spam_like():
    r = list(to_records_spamassassin([SPAMASSASSIN_FIXTURE[1]]))[0]
    labels = r.question.option_labels()
    assert r.target.label_index == labels.index("legitimate")


def test_spamassassin_hard_ham_maps_to_spam_like_category_not_plain_legitimate():
    r = list(to_records_spamassassin([SPAMASSASSIN_FIXTURE[2]]))[0]
    labels = r.question.option_labels()
    idx = r.target.label_index
    assert labels[idx] == "legitimate_spam_like"
    assert labels[idx] != "legitimate"


def test_spamassassin_three_categories_are_mutually_exclusive_one_hot():
    for folder, subject, body in SPAMASSASSIN_FIXTURE:
        r = list(to_records_spamassassin([(folder, subject, body)]))[0]
        assert sum(r.target.distribution) == 1.0
        assert r.target.distribution.count(1.0) == 1


def test_spamassassin_state_includes_subject_and_body():
    r = list(to_records_spamassassin([SPAMASSASSIN_FIXTURE[0]]))[0]
    assert "Buy now!!!" in r.state
    assert "free money" in r.state


def test_extract_subject_and_body_parses_simple_rfc822_message():
    raw = (
        b"From: sender@example.com\r\n"
        b"Subject: Hello there\r\n"
        b"Content-Type: text/plain\r\n\r\n"
        b"This is the body.\r\n"
    )
    result = _extract_subject_and_body(raw)
    assert result == ("Hello there", "This is the body.")


def test_extract_subject_and_body_returns_none_for_empty_body():
    raw = b"From: sender@example.com\r\nSubject: Empty\r\n\r\n"
    assert _extract_subject_and_body(raw) is None
