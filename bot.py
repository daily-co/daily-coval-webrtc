#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import os

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecatcloud.agent import DailySessionArguments

from pipecat.frames.frames import (
    EndFrame,
    ErrorFrame,
    TTSSpeakFrame,
    EndTaskFrame,
    BotStartedSpeakingFrame,
    UserStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    UserStoppedSpeakingFrame,
    TTSTextFrame,
    LLMFullResponseStartFrame,
    TransportMessageUrgentFrame,
    TransportMessageFrame
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.pipeline.task import (
    PipelineParams,
    PipelineTask,
    PipelineTaskSink,
    PipelineTaskSource,
)

from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor

# Check if we're in local development mode
LOCAL_RUN = os.getenv("LOCAL_RUN")
if LOCAL_RUN:
    import asyncio
    import webbrowser

    try:
        from local_runner import configure
    except ImportError:
        logger.error(
            "Could not import local_runner module. Local development mode may not work."
        )

# Load environment variables
load_dotenv(override=True)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class CustomObserver(BaseObserver):
    async def on_push_frame(self, data: FramePushed):
        src = data.source
        dst = data.destination
        frame = data.frame
        direction = data.direction
        timestamp = data.timestamp

        # Create direction arrow
        arrow = "→" if direction == FrameDirection.DOWNSTREAM else "←"

        color_start = (
            bcolors.OKGREEN
            if direction == FrameDirection.DOWNSTREAM
            else bcolors.OKBLUE
        )

        if isinstance(dst, PipelineTaskSink) or isinstance(dst, PipelineTaskSource):
            if isinstance(frame, BotStartedSpeakingFrame):
                print(
                    f"🤖 {color_start} start {frame}: {src} {arrow} {dst} {bcolors.ENDC}"
                )
            elif isinstance(frame, BotStoppedSpeakingFrame):
                print(
                    f"🤖 {color_start} STOP {frame}: {src} {arrow} {dst} {bcolors.ENDC}"
                )
            elif isinstance(frame, UserStartedSpeakingFrame):
                print(
                    f"😳 {color_start} start {frame}: {src} {arrow} {dst} {bcolors.ENDC}"
                )
            elif isinstance(frame, UserStoppedSpeakingFrame):
                print(
                    f"😳 {color_start} STOP {frame}: {src} {arrow} {dst} {bcolors.ENDC}"
                )

            elif isinstance(frame, EndFrame):
                print(f"📌 {color_start} {frame}: {src} {arrow} {dst} {bcolors.ENDC}")
            elif isinstance(frame, EndTaskFrame):
                print(f"🔥 {color_start} {frame}: {src} {arrow} {dst} {bcolors.ENDC}")
            elif isinstance(frame, ErrorFrame):
                print(f"💔😭 {color_start} {frame}: {src} {arrow} {dst} {bcolors.ENDC}")
            elif isinstance(frame, LLMFullResponseStartFrame):
                print(f"🧠 {color_start} {frame}: {src} {arrow} {dst} {bcolors.ENDC}")
            elif isinstance(frame, TransportMessageUrgentFrame):
                print(f"😈 {color_start} {frame}: {src} {arrow} {dst} {bcolors.ENDC}")
            elif isinstance(frame, TransportMessageFrame):
                print(f"😈 {color_start} {frame}: {src} {arrow} {dst} {bcolors.ENDC}")


async def main(room_url: str, token: str):
    """Main pipeline setup and execution function.

    Args:
        room_url: The Daily room URL
        token: The Daily room token
    """
    logger.debug("Starting bot in room: {}", room_url)

    transport = DailyTransport(
        room_url,
        token,
        "bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # transcription_enabled=True, ####### use DEEPGRAM instead
        ),
    )

    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY")) #######

    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",
    )

    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful LLM in a WebRTC call. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way. Answer the user's questions. You know a lot about laserdiscs. Always start with 'Aloha'",
        },
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            stt, #######
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
            # report_only_initial_ttfb=True,
        ),
        observers=[
            CustomObserver(),
            RTVIObserver(rtvi),
        ],
    )
    participant_joined = False

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        logger.debug("Client ready event received")
        await rtvi.set_bot_ready()


    # @transport.event_handler("on_first_participant_joined")
    # async def on_first_participant_joined(transport, participant):
    #     logger.info("First participant joined: {}", participant["id"])
    #     # Capture the participant's transcription
    #     await transport.capture_participant_transcription(participant["id"])
    #     # Kick off the conversation
    #     await task.queue_frames([context_aggregator.user().get_context_frame()])


    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        logger.info("pretend we heard a client-ready event")
        await rtvi.set_bot_ready()
        participant_joined = True

        logger.info("💕💕 First participant joined: {}", participant["id"])
        await transport.capture_participant_transcription(participant["id"])
        # Kick off the conversation.
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        if participant_joined:
            logger.debug("pretend we heard a client-ready event")
            await rtvi.set_bot_ready()

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
    logger.info(f"👾Bot process initialized {args.room_url} {args.token}")

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


# Local development entry point
if LOCAL_RUN and __name__ == "__main__":
    try:
        asyncio.run(local_main())
    except Exception as e:
        logger.exception(f"Failed to run in local mode: {e}")
