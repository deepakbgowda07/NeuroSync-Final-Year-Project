"""Tests for utils.error_handling."""

import pytest

from utils.error_handling import (
    CameraError,
    CorruptedFileError,
    CUDAUnavailableError,
    DatasetError,
    InvalidConfigurationError,
    LowMemoryError,
    ModelWeightsError,
    StrokeRehabAIError,
    check_available_memory_mb,
    format_user_facing_message,
    handle_gracefully,
)


def test_all_exception_types_are_strokerehabai_errors():
    for exc_cls in (CameraError, CUDAUnavailableError, DatasetError, ModelWeightsError,
                    CorruptedFileError, InvalidConfigurationError, LowMemoryError):
        assert issubclass(exc_cls, StrokeRehabAIError)


def test_custom_error_carries_user_message():
    exc = CameraError("Camera disconnected.", detail="cv2.VideoCapture returned False")
    assert exc.user_message == "Camera disconnected."
    assert "VideoCapture" in exc.detail


def test_format_user_facing_message_uses_custom_message():
    exc = DatasetError("Dataset not found. Please download it first.")
    assert format_user_facing_message(exc) == "Dataset not found. Please download it first."


def test_format_user_facing_message_fallback_for_standard_exceptions():
    assert "could not be found" in format_user_facing_message(FileNotFoundError()).lower()
    assert "memory" in format_user_facing_message(MemoryError()).lower()


def test_format_user_facing_message_generic_fallback():
    class WeirdError(Exception):
        pass
    message = format_user_facing_message(WeirdError())
    assert "unexpected error" in message.lower()


def test_handle_gracefully_returns_default_on_exception():
    @handle_gracefully(default_return="fallback")
    def always_fails():
        raise ValueError("boom")

    assert always_fails() == "fallback"


def test_handle_gracefully_reraises_as_strokerehabai_error():
    @handle_gracefully(reraise=True, context="test")
    def always_fails():
        raise ValueError("boom")

    with pytest.raises(StrokeRehabAIError):
        always_fails()


def test_handle_gracefully_lets_strokerehabai_errors_pass_through_unchanged():
    @handle_gracefully(reraise=True)
    def raises_custom():
        raise CameraError("Specific camera message.")

    with pytest.raises(CameraError, match="Specific camera message."):
        raises_custom()


def test_handle_gracefully_success_path_returns_normally():
    @handle_gracefully(default_return=None)
    def works_fine():
        return 42

    assert works_fine() == 42


def test_check_available_memory_mb_does_not_raise_without_cuda():
    check_available_memory_mb(min_required_mb=500.0)  # CPU sandbox: should just return
