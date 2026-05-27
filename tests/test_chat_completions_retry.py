import asyncio
import json
import unittest
from unittest.mock import patch

from mimo2api import web_service


class FakeRequest:
    method = "POST"

    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    async def body(self) -> bytes:
        return self._body


class ChatCompletionsRetryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.web_service = web_service
        self.original_active_clients = list(web_service.state.active_clients)
        self.original_client_cooldowns = dict(web_service.state.client_cooldowns)
        web_service.state.active_clients = [object(), object()]
        web_service.state.client_cooldowns = {}

    async def asyncTearDown(self):
        self.web_service.state.active_clients = self.original_active_clients
        self.web_service.state.client_cooldowns = self.original_client_cooldowns
        self.web_service.state.pending_queues.clear()
        self.web_service.state.req_id_timestamps.clear()
        self.web_service.state.req_id_to_ws_id.clear()
        self.web_service.state.ws_to_req_ids.clear()

    async def test_non_streaming_chat_retries_when_node_disconnects_after_start(self):
        first_queue = asyncio.Queue()
        await first_queue.put({"type": "error", "body": "节点断开连接"})
        second_queue = asyncio.Queue()
        await second_queue.put({"type": "chunk", "body": '{"id":"ok","choices":[]}'})
        await second_queue.put({"type": "finish"})

        attempts = [
            self.web_service.ForwardAttempt(
                req_id="first",
                queue=first_queue,
                target_ws=object(),
                first_msg={"type": "start", "status": 200, "headers": {"content-type": "application/json"}},
                attempt_number=1,
            ),
            self.web_service.ForwardAttempt(
                req_id="second",
                queue=second_queue,
                target_ws=object(),
                first_msg={"type": "start", "status": 200, "headers": {"content-type": "application/json"}},
                attempt_number=2,
            ),
        ]

        async def fake_prepare_forward_attempt(**kwargs):
            return attempts.pop(0)

        request = FakeRequest({"model": "mimo-v2.5", "messages": [{"role": "user", "content": "hi"}]})

        with (
            patch.object(self.web_service, "prepare_forward_attempt", side_effect=fake_prepare_forward_attempt),
            patch.object(self.web_service, "record_request_started"),
            patch.object(self.web_service, "record_request_finished"),
        ):
            response = await self.web_service._forward_request(request, "/v1/chat/completions")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(getattr(response, "body", None), b'{"id":"ok","choices":[]}')
        self.assertEqual(len(attempts), 0)

    async def test_streaming_chat_retries_when_node_disconnects_before_first_chunk(self):
        first_queue = asyncio.Queue()
        await first_queue.put({"type": "error", "body": "节点断开连接"})
        second_queue = asyncio.Queue()
        await second_queue.put({"type": "chunk", "body": "data: second\n\n"})
        await second_queue.put({"type": "finish"})

        attempts = [
            self.web_service.ForwardAttempt(
                req_id="first-stream",
                queue=first_queue,
                target_ws=object(),
                first_msg={"type": "start", "status": 200, "headers": {"content-type": "text/event-stream"}},
                attempt_number=1,
            ),
            self.web_service.ForwardAttempt(
                req_id="second-stream",
                queue=second_queue,
                target_ws=object(),
                first_msg={"type": "start", "status": 200, "headers": {"content-type": "text/event-stream"}},
                attempt_number=2,
            ),
        ]

        async def fake_prepare_forward_attempt(**kwargs):
            return attempts.pop(0)

        request = FakeRequest({"model": "mimo-v2.5", "stream": True, "messages": [{"role": "user", "content": "hi"}]})

        with (
            patch.object(self.web_service, "prepare_forward_attempt", side_effect=fake_prepare_forward_attempt),
            patch.object(self.web_service, "record_request_started"),
            patch.object(self.web_service, "record_request_finished"),
            patch.object(self.web_service, "STREAM_KEEPALIVE_INTERVAL", 0.001),
            patch.object(self.web_service, "STREAM_CHUNK_TIMEOUT", 0),
        ):
            response = await self.web_service._forward_request(request, "/v1/chat/completions")
            body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body, b"data: second\n\n")
        self.assertEqual(len(attempts), 0)

    async def test_streaming_chat_emits_error_when_node_disconnects_after_first_chunk(self):
        queue = asyncio.Queue()
        await queue.put({"type": "chunk", "body": "data: first\n\n"})
        await queue.put({"type": "error", "body": "节点断开连接"})

        attempts = [
            self.web_service.ForwardAttempt(
                req_id="stream",
                queue=queue,
                target_ws=object(),
                first_msg={"type": "start", "status": 200, "headers": {"content-type": "text/event-stream"}},
                attempt_number=1,
            )
        ]

        async def fake_prepare_forward_attempt(**kwargs):
            return attempts.pop(0)

        request = FakeRequest({"model": "mimo-v2.5", "stream": True, "messages": [{"role": "user", "content": "hi"}]})

        with (
            patch.object(self.web_service, "prepare_forward_attempt", side_effect=fake_prepare_forward_attempt),
            patch.object(self.web_service, "record_request_started"),
            patch.object(self.web_service, "record_request_finished"),
            patch.object(self.web_service, "STREAM_KEEPALIVE_INTERVAL", 0.001),
            patch.object(self.web_service, "STREAM_CHUNK_TIMEOUT", 0),
        ):
            response = await self.web_service._forward_request(request, "/v1/chat/completions")
            body = b"".join([chunk async for chunk in response.body_iterator])

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"data: first\n\n", body)
        self.assertIn(b"event: error", body)
        self.assertIn("节点断开连接".encode("utf-8"), body)
        self.assertEqual(len(attempts), 0)


if __name__ == "__main__":
    unittest.main()
