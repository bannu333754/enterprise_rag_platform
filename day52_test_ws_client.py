import asyncio
import websockets
import json
import sys

async def run_client():
    session_id = "test_user_01"
    uri = f"ws://127.0.0.1:8000/ws/chat/{session_id}"

    question = "What are the core failure recovery mechanisms described in the documents?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    print(f"\n[Connecting to WebSocket: {uri}]")
    async with websockets.connect(uri) as websocket:
        print("[Connected! Transmitting query frame...]")
        payload = {
            "question": question,
            "target_doc": "ALL"
        }
        await websocket.send(json.dumps(payload))

        while True:
            raw_msg = await websocket.recv()
            data = json.loads(raw_msg)
            msg_type = data.get("type")

            if msg_type == "event":
                print(f"\n[Telemetry Event: {data.get('stage')}] -> {data.get('message')}")
            elif msg_type == "citations":
                print("\n--- CITATIONS DISCOVERED ---")
                for c in data.get("data", []):
                    print(f"• [{c['source']} p.{c['page']}] (Cross-Score: {c['score']})")
                print("----------------------------\n[Assistant Response]: ", end="", flush=True)
            elif msg_type == "token":
                print(data.get("delta"), end="", flush=True)
            elif msg_type == "done":
                print(f"\n\n[Stream Completed | Session: {data.get('session_id')}]")
                break
            elif msg_type == "error":
                print(f"\n[Error Occurred]: {data.get('message')}")
                break

if __name__ == "__main__":
    asyncio.run(run_client())
