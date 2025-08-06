import aiohttp
import asyncio
import json
import random

SERVER_URL = "http://localhost:8000"  # Change as needed

async def send_prompt(prompt: str):
    payload = {
        "lora_dir": "tloen/alpaca-lora-7b",  # Adjust as needed
        "inputs": prompt,
        "parameters": {
            "do_sample": False,
            "ignore_eos": True,
            "max_new_tokens": 8,
        },
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{SERVER_URL}/generate", json=payload) as resp:
                body = await resp.read()
                try:
                    data = json.loads(body)
                    generated_text = data.get("generated_text", [""])[0]
                    feedback_id = data.get("feedback", None)
                except json.JSONDecodeError:
                    print("Failed to decode JSON response.")
                    generated_text = body.decode(errors="replace")
                    feedback_id = None
    except aiohttp.ClientError as e:
        print(f"⚠️ Request failed: {e}")
        return "<request failed>", None

    return generated_text, feedback_id

async def send_feedback(req_id: str, label: int):
    payload = {
        "req_id": req_id,
        "label": label,
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{SERVER_URL}/feedback", json=payload) as resp:
                result = await resp.text()
                print(f"[Feedback response] {result.strip()}")
    except aiohttp.ClientError as e:
        print(f"Feedback request failed: {e}")

async def chat():
    print("💬 Completion client started. Type 'quit' to exit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in {"quit", "exit"}:
            print("Exiting chat.")
            break

        generated_text, feedback_id = await send_prompt(user_input)
        print(f"Generated Completion: {generated_text}")

        if feedback_id:
            while True:
                feedback_input = input("\033[92mDo you like the completion? (1 / -1): \033[0m").strip()
                if feedback_input in {"1", "-1"}:
                    await send_feedback(feedback_id, int(feedback_input))
                    break
                else:
                    print("Invalid input. Please enter 1 or -1.")

if __name__ == "__main__":
    try:
        asyncio.run(chat())
    except KeyboardInterrupt:
        print("\nInterrupted — exiting…")