"""Cron job execution for the gateway runtime."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from loguru import logger

import nanobot.utils.evaluator as evaluator
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.message import MessageTool
from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.cron.types import CronJob


class DeliverToChannel(Protocol):
    def __call__(
        self,
        msg: OutboundMessage,
        *,
        record: bool = False,
        session_key: str | None = None,
    ) -> Awaitable[None]: ...


ChannelLookup = Callable[[str], Any | None]


class _CronStreamBuffer:
    def __init__(
        self,
        *,
        channel: str,
        chat_id: str,
        channel_meta: dict[str, Any],
        base_id: str,
    ) -> None:
        self.channel = channel
        self.chat_id = chat_id
        self.channel_meta = channel_meta
        self.base_id = base_id
        self.segment = 0
        self.events: list[OutboundMessage] = []
        self.has_delta = False

    def _stream_id(self) -> str:
        return f"{self.base_id}:{self.segment}"

    async def on_stream(self, delta: str) -> None:
        meta = dict(self.channel_meta)
        meta["_stream_delta"] = True
        meta["_stream_id"] = self._stream_id()
        self.events.append(OutboundMessage(
            channel=self.channel,
            chat_id=self.chat_id,
            content=delta,
            metadata=meta,
        ))
        if delta:
            self.has_delta = True

    async def on_stream_end(self, *, resuming: bool = False) -> None:
        meta = dict(self.channel_meta)
        meta["_stream_end"] = True
        meta["_resuming"] = resuming
        meta["_stream_id"] = self._stream_id()
        self.events.append(OutboundMessage(
            channel=self.channel,
            chat_id=self.chat_id,
            content="",
            metadata=meta,
        ))
        self.segment += 1

    async def publish(self, bus: MessageBus) -> None:
        for event in self.events:
            await bus.publish_outbound(event)


class CronJobExecutor:
    """Runs scheduled cron jobs through the agent and optional channel delivery."""

    def __init__(
        self,
        *,
        agent: Any,
        bus: MessageBus,
        deliver_to_channel: DeliverToChannel,
        get_channel: ChannelLookup | None = None,
    ) -> None:
        self.agent = agent
        self.bus = bus
        self.deliver_to_channel = deliver_to_channel
        self.get_channel = get_channel or (lambda _channel: None)

    async def run(self, job: CronJob) -> str | None:
        if job.name == "dream":
            try:
                await self.agent.dream.run()
                logger.info("Dream cron job completed")
            except Exception:
                logger.exception("Dream cron job failed")
            return None

        return await self._run_agent_turn(job)

    async def _run_agent_turn(self, job: CronJob) -> str | None:
        reminder_note = self._reminder_note(job)
        cron_tool = self._tool("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            cron_token = cron_tool.set_cron_context(True)

        message_tool = self._tool("message")
        message_record_token = None
        if isinstance(message_tool, MessageTool):
            message_record_token = message_tool.set_record_channel_delivery(True)

        channel_name = job.payload.channel or "cli"
        chat_id = job.payload.to or "direct"
        stream = self._stream_buffer(job, channel_name=channel_name, chat_id=chat_id)

        try:
            resp = await self.agent.process_direct(
                reminder_note,
                session_key=f"cron:{job.id}",
                channel=channel_name,
                chat_id=chat_id,
                on_progress=self._silent,
                on_stream=stream.on_stream if stream else None,
                on_stream_end=stream.on_stream_end if stream else None,
            )
        finally:
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)
            if isinstance(message_tool, MessageTool) and message_record_token is not None:
                message_tool.reset_record_channel_delivery(message_record_token)

        response = resp.content if resp else ""

        if job.payload.deliver and isinstance(message_tool, MessageTool) and message_tool._sent_in_turn:
            await self._publish_turn_end_if_needed(job, channel_name=channel_name, chat_id=chat_id)
            return response

        delivered = False
        if job.payload.deliver and job.payload.to and response:
            should_notify = await evaluator.evaluate_response(
                response, reminder_note, self.agent.provider, self.agent.model,
            )
            if should_notify:
                meta = dict(job.payload.channel_meta)
                if stream and stream.has_delta:
                    await stream.publish(self.bus)
                    meta["_streamed"] = True
                await self.deliver_to_channel(
                    OutboundMessage(
                        channel=channel_name,
                        chat_id=chat_id,
                        content=response,
                        metadata=meta,
                    ),
                    record=True,
                    session_key=job.payload.session_key,
                )
                delivered = True

        if delivered:
            await self._publish_turn_end_if_needed(job, channel_name=channel_name, chat_id=chat_id)
        return response

    def _tool(self, name: str) -> Any | None:
        tools = getattr(self.agent, "tools", {})
        if hasattr(tools, "get"):
            return tools.get(name)
        return None

    def _stream_buffer(
        self,
        job: CronJob,
        *,
        channel_name: str,
        chat_id: str,
    ) -> _CronStreamBuffer | None:
        target_channel = self.get_channel(channel_name)
        wants_stream = bool(
            job.payload.deliver
            and job.payload.to
            and target_channel is not None
            and target_channel.supports_streaming
        )
        if not wants_stream:
            return None
        return _CronStreamBuffer(
            channel=channel_name,
            chat_id=chat_id,
            channel_meta=job.payload.channel_meta,
            base_id=f"cron:{job.id}:{time.time_ns()}",
        )

    async def _publish_turn_end_if_needed(
        self,
        job: CronJob,
        *,
        channel_name: str,
        chat_id: str,
    ) -> None:
        if channel_name != "websocket" or not job.payload.to:
            return
        await self.bus.publish_outbound(OutboundMessage(
            channel=channel_name,
            chat_id=chat_id,
            content="",
            metadata={**job.payload.channel_meta, "_turn_end": True},
        ))

    @staticmethod
    async def _silent(*_args: Any, **_kwargs: Any) -> None:
        pass

    @staticmethod
    def _reminder_note(job: CronJob) -> str:
        return (
            "The scheduled time has arrived. Deliver this reminder to the user now, "
            "as a brief and natural message in their language. Speak directly to them — "
            "do not narrate progress, summarize, include user IDs, or add status reports "
            "like 'Done' or 'Reminded'.\n\n"
            f"Reminder: {job.payload.message}"
        )
