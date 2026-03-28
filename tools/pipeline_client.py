import asyncio
import sys
import warnings
import argparse
import httpx

from agent.orchestrator import VoiceClientOrchestrator
from shared.config import load_config
from shared.logging import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging(__name__)

BASE_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws"


async def check_health():
    async with httpx.AsyncClient() as client:
        try:
            h_resp = await client.get(f"{BASE_URL}/health")
            p_resp = await client.get(f"{BASE_URL}/pipeline/health")
            logger.info(f"Service Health: {h_resp.json()}")
            logger.info(f"Pipeline Health: {p_resp.json()}")
        except Exception as e:
            logger.error(f"Failed to reach health endpoints: {e}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--health", action="store_true")
    args = parser.parse_args()

    if args.health:
        await check_health()
        return

    cfg = load_config()
    orchestrator = VoiceClientOrchestrator(cfg, WS_URL)
    await orchestrator.connect_and_stream()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        sys.exit(0)
