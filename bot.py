#
# Copyright (c) 2025, Daily
#
# SPDXLicenseIdentifier: BSD 2Clause License
#

import aiohttp
import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import DailySessionArguments

# from local_runner import configure

load_dotenv(override=True)


async def main(room_url: str, token: str):
    logger.debug("Starting bot in room: {}", room_url)

    transport = DailyTransport(
        room_url,
        token,
        "Pipecat Voice AI Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    # Configure your STT, LLM, and TTS services here
    # Swap out different processors or properties to customize your bot
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4.1-mini")
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="f31cc6a7-c1e8-4764-980c-60a361443dd1",
    )

    # Set up the initial context for the conversation
    # You can specified initial system and assistant messages here
    messages = [
        {
            "role": "system",
            "content": "You are a customer service agent named Paulie.  First ask if there is anything you can help with, quickly and succinctly.  If they ask about the temperature, tell them it is fifteen degrees.",
        },
    ]

    # This sets up the LLM context by providing messages and tools
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[])) ####
    rtvi.set_errors_enabled(False) #### this _might_ be the secret sauce

    # A core voice AI pipeline
    # Add additional processors to customize the bot's behavior
    pipeline = Pipeline(
        [
            transport.input(),
            rtvi, ####
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            # enable_metrics=True,
            # enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)], ####
    )

    @rtvi.event_handler("on_client_ready") ####
    async def on_client_ready(rtvi):
        logger.debug("Client ready event received")
        await rtvi.set_bot_ready()


    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("💕💕 First participant joined: {}", participant["id"])
        # Capture the participant's transcription
        await transport.capture_participant_transcription(participant["id"])
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        logger.info("💔 Participant left: {}", participant)
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False, force_gc=True)

    await runner.run(task)


async def bot(args: DailySessionArguments):
    """Main bot entry point compatible with the FastAPI route handler.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
        body: The configuration object from the request body
        session_id: The session ID for logging
    """
    logger.info(f"RTVI with errors turned off 👾 Bot process initialized {args.room_url} {args.token}")

    try:
        await main(args.room_url, args.token)
        logger.info("Bot process completed")
    except Exception as e:
        logger.exception(f"Error in bot process: {str(e)}")
        raise

# Local development functions
async def local_main():
    """Function for local development testing."""
    try:
        async with aiohttp.ClientSession() as session:
            (room_url, token) = await configure(session)
            print(f"_____bot.py * room_url: {room_url}")
            logger.warning("_")
            logger.warning("_")
            logger.warning(f"Talk to your voice agent here: {room_url}")
            logger.warning("_")
            logger.warning("_")
            # webbrowser.open(room_url)
            await main(room_url, token)
    except Exception as e:
        logger.exception(f"Error in local development mode: {e}")


# # Local development entry point
# if LOCAL_RUN and __name__ == "__main__":
#     try:
#         asyncio.run(local_main())
#     except Exception as e:
#         logger.exception(f"Failed to run in local mode: {e}")