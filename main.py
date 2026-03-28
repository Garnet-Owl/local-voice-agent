import numpy as np
from agent.orchestrator import VoiceAgentOrchestrator


async def get_health_status(orchestrator: VoiceAgentOrchestrator):
    if orchestrator:
        status = await orchestrator.check_pipeline()
        return {
            "status": "ok" if all(status.values()) else "degraded",
            "details": status,
        }
    return {"status": "initializing"}


async def get_pipeline_health(orchestrator: VoiceAgentOrchestrator):
    if orchestrator:
        dummy_audio = np.random.uniform(-0.01, 0.01, 16000).astype(np.float32)
        success = await orchestrator.run_pipeline_test(dummy_audio)
        return {"status": "healthy" if success else "unhealthy"}
    return {"status": "initializing"}
