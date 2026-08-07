"""The INDIRECT elicitation track: ask the model to write an opinion piece about a
statement, never showing it a scale. The stance is read back out of the prose later
by the cross-encoder (agreement_scoring.py).

Columns written per (language, prompt variant): original_text_<lv>, task_<lv>_v<j>
(the writing brief used), answer_<lv>_v<j>. A cell that never produced content is
left empty rather than marked, since there is no reason field to carry a marker.

The run mechanics (fan-out, checkpointing, --patch, --overwrite_changed) live in
generation.py and are shared with the direct survey track.
"""
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils import call_api
from answer_generation.generation import Track, main

MAX_RETRIES = 5


def load_units(path: str) -> dict[str, list]:
    """{language: [task, ...]} -- one unit per writing brief."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def call_speech(statement, unit, model):
    task = unit
    prompt = f'{task}\n"{statement}"\n'
    for attempt in range(MAX_RETRIES):
        try:
            # Thinking off: the budget is for the speech, not an internal deliberation
            # that can exhaust max_tokens before any content is emitted.
            response = call_api(prompt, model, enable_thinking=False)
            choice = response["choices"][0]
            content = choice["message"]["content"]
            if content:
                return {"task": task, "answer": content}
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                raise ValueError(
                    "Truncated before any content (finish_reason=length): the token budget "
                    "was exhausted (likely by reasoning). Disable thinking or raise max_tokens.")
            raise ValueError(f"API returned empty content (finish_reason={finish_reason}), "
                             f"full message: {choice['message']}")
        except Exception as error:
            snippet = statement[:80].replace("\n", " ")
            print(f"Error (attempt {attempt + 1}/{MAX_RETRIES}) [{model}] "
                  f"stmt={snippet!r}: {error}", flush=True)
        time.sleep(min(2 ** (attempt + 1), 10))
    return {"task": task, "answer": None}


SPEECH_TRACK = Track(
    name="speeches (indirect prose)",
    filename="speeches_{langs}{variant}.csv",
    primary="answer",
    load_units=load_units,
    call=call_speech,
    statements_file=lambda variant: "statements.jsonl" if variant == "" else "statements_negated.jsonl",
    default_workers=4,
    default_variant="_negated",
    default_prompts="generate_speeches.json",
    failure_noun="failed",
)


if __name__ == "__main__":
    main(SPEECH_TRACK)
