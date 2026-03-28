from shared.logging import setup_logging

logger = setup_logging("interrupt_handler")


class InterruptHandler:
    """Detects and handles user barge-in (interrupting the agent)."""

    def __init__(self) -> None:
        self._is_agent_speaking = False
        self._interrupted = False

    def start_agent_speech(self):
        self._is_agent_speaking = True
        self._interrupted = False

    def stop_agent_speech(self):
        self._is_agent_speaking = False

    def check_for_interrupt(self, is_speech_detected: bool) -> bool:
        if self._is_agent_speaking and is_speech_detected:
            logger.info("Interrupt detected (barge-in).")
            self._interrupted = True
            return True
        return False

    def is_interrupted(self) -> bool:
        return self._interrupted

    def clear_interrupt(self):
        self._interrupted = False
        self._is_agent_speaking = False
