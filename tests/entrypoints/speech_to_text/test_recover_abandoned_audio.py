# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit and integration-style tests for recovering audio abandoned in a chunk.

The numbers in these tests are measurements taken from a running
whisper-large-v3-turbo deployment (see the issue this change fixes), not values
chosen to make the assertions pass: a fixture invented from memory would exercise
a server that does not exist.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from numpy import zeros

from vllm.entrypoints.speech_to_text.base.serving import (
    SPARSE_MIN_DURATION_S,
    SpeechToTextBaseServing,
    is_sparse_recovery,
    last_decoded_timestamp,
    strip_seam_overlap,
)
from vllm.entrypoints.speech_to_text.transcription.protocol import (
    TranscriptionResponseVerbose,
)
from vllm.sampling_params import SamplingParams


class TestLastDecodedTimestamp:
    """Whisper states how far it got through its timestamp tokens."""

    def test_reads_the_final_timestamp(self):
        init = 50365  # any base; only the offset from it matters
        # <|0.00|> text <|5.82|> <|5.82|> text <|21.56|>
        tokens = (init, 1, 2, init + 291, init + 291, 3, 4, init + 1078)
        assert last_decoded_timestamp(tokens, init) == pytest.approx(21.56)

    def test_returns_none_without_timestamps(self):
        assert last_decoded_timestamp((1, 2, 3), 50365) is None


class TestIsSparseRecovery:
    """A few words spread over many seconds are not speech, whatever they say."""

    @pytest.mark.parametrize(
        "text,duration",
        [
            (" Thank you.", 13.32),  # measured phantom over silence
            (" Let's go.", 13.82),  # measured, and in no blocklist anywhere
        ],
    )
    def test_drops_measured_phantoms(self, text, duration):
        assert is_sparse_recovery(text, duration) is True

    def test_keeps_speech_at_a_real_speaking_rate(self):
        speech = (
            "the first words I spoke in the original phonograph a little piece "
            "of practical poetry mary had a little lamb its fleece was white as snow"
        )
        assert is_sparse_recovery(speech, 15.1) is False

    def test_does_not_judge_a_short_region_by_density(self):
        assert is_sparse_recovery("done", SPARSE_MIN_DURATION_S - 0.1) is False

    def test_counts_characters_for_no_space_languages(self):
        assert (
            is_sparse_recovery(
                "这是一个完整句子",
                4.0,
                language="zh",
                no_space_languages={"zh", "ja"},
            )
            is False
        )

    def test_drops_sparse_no_space_language_recovery(self):
        assert (
            is_sparse_recovery(
                "谢谢",
                13.82,
                language="zh",
                no_space_languages={"zh", "ja"},
            )
            is True
        )


class TestStripSeamOverlap:
    """The pre-roll buys context at the price of a repeated word or two."""

    def test_removes_the_repetition_the_preroll_caused(self):
        assert strip_seam_overlap("and then the kitchen", "the kitchen has a stove") == (
            "has a stove"
        )

    def test_ignores_case_and_punctuation(self):
        assert strip_seam_overlap("finishing up", "Finishing up, that is all") == "that is all"

    def test_leaves_a_single_shared_word_alone(self):
        # Speech shares words all the time; one match is not evidence.
        assert strip_seam_overlap("twenty seconds", "seconds later we start") == (
            "seconds later we start"
        )

    def test_leaves_a_long_repetition_alone(self):
        # Half a second cannot have produced five words: this is the speaker
        # repeating himself, and the transcript keeps what he said.
        assert strip_seam_overlap("one two three four five", "one two three four five six") == (
            "one two three four five six"
        )


class _TimestampTokenizer:
    eos_token_id = 900

    def encode(self, text, add_special_tokens=False):
        assert text == "<|0.00|>"
        return [1000]

    def decode(self, token_ids):
        return "".join(
            {10: "first", 11: "tail one two three"}[token] for token in token_ids
        )


def _logprobs(tokens):
    return [{token: SimpleNamespace(logprob=-0.1)} for token in tokens]


def _output(tokens):
    completion = SimpleNamespace(
        token_ids=tokens,
        logprobs=_logprobs(tokens),
        text="",
    )
    return SimpleNamespace(outputs=[completion], finished=True)


async def _result_generator(output):
    yield output


@pytest.mark.asyncio
async def test_recovery_preserves_tail_and_monotonic_segments():
    """An early EOT recovers its tail without overlapping segment timestamps."""
    first_end_token = 1000 + round(26.4 / 0.02)
    tail_end_token = 1000 + round(4.1 / 0.02)
    outputs = [
        _output((10, first_end_token, first_end_token, 900)),
        _output((11, tail_end_token, tail_end_token, 900)),
    ]

    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving._check_model = AsyncMock(return_value=None)
    serving._base_request_id = Mock(return_value="transcribe-test")
    serving._maybe_get_adapters = Mock(return_value=None)
    serving._log_inputs = Mock()
    serving._parse_stt_prompt = Mock(return_value={"prompt": "recovery"})
    serving.asr_config = SimpleNamespace(max_audio_clip_s=30, sample_rate=16_000)
    serving.default_sampling_params = {}
    serving.model_config = SimpleNamespace(max_model_len=100)
    serving.task_type = "transcribe"
    serving.tokenizer = _TimestampTokenizer()
    serving.model_cls = SimpleNamespace(
        no_space_languages=set(), supports_segment_timestamp=True
    )
    serving._preprocess_speech_to_text = AsyncMock(
        return_value=(
            [{"prompt": "chunk"}],
            30.0,
            [0.0],
            [zeros(30 * 16_000)],
        )
    )
    serving.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(return_value=[{"prompt": "recovery"}])
    )
    serving.engine_client = SimpleNamespace(
        errored=False,
        generate=Mock(side_effect=[_result_generator(output) for output in outputs]),
    )

    request = SimpleNamespace(
        model="whisper",
        language="en",
        response_format="verbose_json",
        stream=False,
        use_beam_search=False,
        max_completion_tokens=None,
        temperature=0.0,
        to_sampling_params=Mock(
            return_value=SamplingParams(max_tokens=32, logprobs=1)
        ),
    )

    response = await serving._create_speech_to_text(
        audio_data=b"audio",
        request=request,
        raw_request=None,
        response_class=TranscriptionResponseVerbose,
        stream_generator_method=Mock(),
    )

    assert isinstance(response, TranscriptionResponseVerbose)
    assert response.text == "first tail one two three"
    assert [(segment.start, segment.end) for segment in response.segments] == [
        pytest.approx((0.0, 26.4)),
        pytest.approx((26.4, 30.0)),
    ]
    assert all(
        previous.end <= current.start
        for previous, current in zip(response.segments, response.segments[1:])
    )
    assert [call.args[2] for call in serving.engine_client.generate.call_args_list] == [
        "transcribe-transcribe-test",
        "transcribe-transcribe-test-recover-0",
    ]
