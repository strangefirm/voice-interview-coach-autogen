import metrics


def test_tracker_sums_tokens_chars_and_time():
    tracker = metrics.Tracker()
    usage = metrics.Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    with tracker.track("llm", usage):
        pass
    with tracker.track("tts", chars=1_000_000):
        pass
    p = metrics.PRICES
    expected = p["input_per_1m_tokens"] + p["output_per_1m_tokens"] + p["tts_per_1m_chars"]
    assert abs(tracker.total_cost - expected) < 1e-9
    assert set(tracker.summary()) == {"llm", "tts"}


def test_failed_call_is_still_timed():
    tracker = metrics.Tracker()
    try:
        with tracker.track("boom"):
            raise ValueError
    except ValueError:
        pass
    assert len(tracker.calls) == 1
