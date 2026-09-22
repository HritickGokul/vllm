# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for timestamped speech-to-text chunking."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest

from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.entrypoints.speech_to_text.base.serving import SpeechToTextBaseServing
from vllm.entrypoints.speech_to_text.transcription.protocol import TranscriptionSegment
from vllm.sampling_params import SamplingParams

SR = 16_000
_PATCH = "vllm.entrypoints.speech_to_text.base.serving"


@pytest.mark.asyncio
async def test_chunk_offsets_are_cumulative_not_nominal():
    """chunk_start_offsets must be cumulative actual chunk lengths, not the old approach
    of 'idx * max_audio_clip_s'.  When split_audio places a boundary before the
    nominal 30 s mark, the old formula drifts; the fixed formula stays exact.
    """
    # Chunks shorter than exactly 30 s, as split_audio produces when a quiet
    # region falls inside the 1 s overlap window before the nominal boundary.
    chunk_lengths = [int(29.5 * SR), int(29.7 * SR), int(5.0 * SR)]
    chunks = [np.zeros(n, dtype=np.float32) for n in chunk_lengths]

    duration = sum(chunk_lengths) / SR

    expected_offsets = [0.0, 29.5, 29.5 + 29.7]  # cumulative seconds
    wrong_offsets = [0.0, 30.0, 60.0]  # what the old bug produced

    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving._decode_and_chunk_speech_async = AsyncMock(return_value=(chunks, duration))
    serving.asr_config = SpeechToTextConfig(
        sample_rate=float(SR),
        max_audio_clip_s=30,
        overlap_chunk_second=1,
        min_energy_split_window_size=1600,
    )
    serving.model_cls = MagicMock()
    serving.model_cls.validate_language.side_effect = lambda lang: lang
    serving.model_cls.supports_explicit_language_detection = False
    serving.model_cls.get_generation_prompt.return_value = {}
    serving.model_config = MagicMock()
    serving.task_type = "transcribe"
    serving.renderer = MagicMock()
    serving.renderer.render_cmpl_async = AsyncMock(
        return_value=[MagicMock()] * len(chunks)
    )

    request = MagicMock()
    request.language = "en"
    request.to_language = None
    request.response_format = "json"
    request.build_stt_params.return_value = MagicMock()

    with patch(f"{_PATCH}.parse_model_prompt", return_value=MagicMock()):
        _, _, offsets = await serving._preprocess_speech_to_text(
            request=request,
            audio_data=b"\x00",
            request_id="test",
        )

    assert offsets == pytest.approx(expected_offsets, abs=1e-6)
    assert offsets != pytest.approx(wrong_offsets, abs=1e-6)


class _TimestampTokenizer:
    eos_token_id = 2000

    def encode(self, text, add_special_tokens=False):
        assert text == "<|0.00|>"
        return [1000]

    def decode(self, token_ids):
        return "".join({10: "first", 11: "tail"}[token] for token in token_ids)


def _logprobs(tokens):
    return [{token: SimpleNamespace(logprob=-0.1)} for token in tokens]


def _output(tokens):
    completion = SimpleNamespace(token_ids=tokens, logprobs=_logprobs(tokens))
    return SimpleNamespace(outputs=[completion], finished=True)


async def _result_generator(output):
    yield output


def test_verbose_json_keeps_text_without_a_closing_timestamp():
    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving.tokenizer = _TimestampTokenizer()
    tokens = (11, 2000)

    segments = serving._get_verbose_segments(
        tokens=tokens,
        log_probs=_logprobs(tokens),
        request=SimpleNamespace(temperature=0.0),
        segment_class=TranscriptionSegment,
        window_duration=3.6,
    )

    assert len(segments) == 1
    assert segments[0].text == "tail"
    assert segments[0].end == pytest.approx(3.6)


@pytest.mark.asyncio
async def test_verbose_json_redecodes_unfinished_window_tail():
    """The next window starts at the last completed timestamp, not at 30 s."""
    first_end_token = 1000 + round(26.4 / 0.02)
    tail_end_token = 1000 + round(3.6 / 0.02)
    outputs = [
        _output((10, first_end_token, first_end_token, 2000)),
        _output((11, tail_end_token, 2000)),
    ]

    serving = SpeechToTextBaseServing.__new__(SpeechToTextBaseServing)
    serving._decode_speech_async = AsyncMock(
        return_value=(np.arange(30 * SR, dtype=np.float32), 30.0)
    )
    serving.asr_config = SpeechToTextConfig(
        sample_rate=float(SR),
        max_audio_clip_s=30,
    )
    serving.model_cls = SimpleNamespace(
        validate_language=lambda language: language,
        supports_explicit_language_detection=False,
        no_space_languages=set(),
    )
    serving.model_config = MagicMock()
    serving.task_type = "transcribe"
    serving.tokenizer = _TimestampTokenizer()
    serving._build_speech_to_text_prompt = Mock(return_value={})
    serving.renderer = SimpleNamespace(
        render_cmpl_async=AsyncMock(return_value=[{"prompt": "chunk"}])
    )
    serving._get_speech_to_text_sampling_params = Mock(
        return_value=SamplingParams(max_tokens=32, logprobs=1)
    )
    serving._log_inputs = Mock()
    serving.engine_client = SimpleNamespace(
        generate=Mock(side_effect=[_result_generator(output) for output in outputs]),
        abort=AsyncMock(),
    )
    request = SimpleNamespace(
        language="en",
        to_language=None,
        response_format="verbose_json",
        temperature=0.0,
    )

    response = await serving._create_seek_based_verbose_response(
        audio_data=b"audio",
        request=request,
        raw_request=None,
        request_id="transcribe-test",
        lora_request=None,
    )

    chunks = [
        call.args[1] for call in serving._build_speech_to_text_prompt.call_args_list
    ]
    assert [chunk.shape[-1] for chunk in chunks] == [30 * SR, round(3.6 * SR)]
    assert chunks[1][0] == round(26.4 * SR)
    assert [call.args[2] for call in serving.engine_client.generate.call_args_list] == [
        "transcribe-test",
        "transcribe-test-1",
    ]
    assert response.text == "first tail"
    assert response.segments is not None
    assert [(segment.start, segment.end) for segment in response.segments] == [
        pytest.approx((0.0, 26.4)),
        pytest.approx((26.4, 30.0)),
    ]
    assert [segment.id for segment in response.segments] == [0, 1]
