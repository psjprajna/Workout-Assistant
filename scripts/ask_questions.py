"""Run a scripted conversation against the running API and print the replies.

Usage: start the server first, then: uv run python scripts/ask_questions.py
Covers the main cases: pick from a list, unknown exercise, no set details, a bad pick.
"""

import httpx

URL = "http://127.0.0.1:8000/chat"

# (session id, message). Same id = same conversation.
SCRIPT = [
    ("s1", "last time I did bench press?"),  # which one?
    ("s1", "2"),                              # pick one -> sets in kg
    ("s2", "last time I did zzz?"),           # not in your data
    ("s3", "last time I did walking?"),       # logged, but no set details
    ("s4", "last time I did ab rollout?"),    # two exercises, same name
    ("s4", "9"),                              # out of range -> asks again
    ("s4", "2"),                              # valid pick
    ("s5", "2"),                              # a number with no question pending
    ("s6", "what's the weather?"),            # out of scope
]


def main() -> None:
    with httpx.Client(timeout=10) as client:
        for session_id, message in SCRIPT:
            response = client.post(URL, json={"session_id": session_id, "message": message})
            reply = response.json()["reply"] if response.status_code == 200 else response.text
            print(f"[{session_id}] you> {message}")
            print(f"[{session_id}] bot> {reply}\n")


if __name__ == "__main__":
    main()