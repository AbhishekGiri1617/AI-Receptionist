from dotenv import load_dotenv

from livekit import agents
from langchain_core.messages import ToolMessage
from livekit.agents import Agent, AgentServer, AgentSession, TurnHandlingOptions, inference
from livekit.plugins import langchain as lk_langchain

from config import settings, business_config
from langchain_agent import agent_graph

_ = load_dotenv()


def _is_tool_token(item: object) -> bool:
    """True for the (ToolMessage, metadata) pairs astream emits from the tools node."""
    return isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], ToolMessage)


class SpeechOnlyGraph:
    """Wraps the graph so tool return values never reach TTS."""

    def __init__(self, graph: object) -> None:
        self._graph = graph

    def astream(self, *args: object, **kwargs: object):
        inner = self._graph.astream(*args, **kwargs)

        async def _filtered():
            async for item in inner:
                if not _is_tool_token(item):
                    yield item

        return _filtered()


class VoiceAssistantAgent(Agent):
    """Configurable voice support agent backed by a generic LangChain agent graph."""

    def __init__(self) -> None:
        company_name = business_config.get("company_name", "the company")
        agent_name = business_config.get("agent_name", "Assistant")
        super().__init__(
            instructions=f"You are {agent_name}, the voice customer assistant for {company_name}.",
            llm=lk_langchain.LLMAdapter(graph=SpeechOnlyGraph(agent_graph)),
        )


from voice_agent.transcript import TranscriptRecorder


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: agents.JobContext):
    keyterms = business_config.get("stt_keyterms", ["Whalexy", "Mobius Bloom"])
    room_name = ctx.room.name if (ctx.room and hasattr(ctx.room, "name")) else "console-room"
    recorder = TranscriptRecorder(room_name=room_name)

    session = AgentSession(
        stt=inference.STT(
            model=settings.stt_model,
            language=settings.stt_language,
            extra_kwargs={"keyterm": keyterms},
        ),
        tts=inference.TTS(
            model=settings.tts_model,
            voice=settings.tts_voice,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "fixed",
                "min_delay": 0.5,
                "max_delay": 6.0,
            },
        ),
    )

    @session.on("conversation_item_added")
    def on_item_added(ev):
        if hasattr(ev, "item") and ev.item:
            recorder.add_message(ev.item.role, ev.item.text_content)

    @session.on("close")
    def on_session_close(ev):
        recorder.save()

    @ctx.add_shutdown_callback
    async def on_shutdown():
        # Ensure any messages in session history are captured
        if hasattr(session, "history") and session.history:
            for msg in session.history.messages:
                recorder.add_message(msg.role, msg.text_content)
        recorder.save()

    await session.start(agent=VoiceAssistantAgent(), room=ctx.room)

    company_name = business_config.get("company_name", "our company")
    agent_name = business_config.get("agent_name", "Assistant")
    raw_greeting = business_config.get(
        "welcome_greeting",
        "Greet the user warmly as {agent_name} from {company_name} and ask how you can help."
    )
    try:
        greeting = raw_greeting.format(company_name=company_name, agent_name=agent_name)
    except Exception:
        greeting = raw_greeting

    await session.generate_reply(user_input=greeting)


if __name__ == "__main__":
    agents.cli.run_app(server)

