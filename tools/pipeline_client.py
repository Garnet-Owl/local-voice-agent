import asyncio
import sys
import warnings

from agent.orchestrator import VoiceClientOrchestrator
from shared.config import load_config
from shared.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging("client_entry")

WS_URL = "ws://127.0.0.1:8000/ws"


async def main():
    cfg = load_config()
    orchestrator = VoiceClientOrchestrator(cfg, WS_URL)
    await orchestrator.connect_and_stream()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
