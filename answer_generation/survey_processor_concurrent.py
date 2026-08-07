"""The DIRECT elicitation track: show the model a statement plus the 1-5 scale and
record the choice it picks along with its written justification.

Columns written per (language, prompt variant): original_text_<lv>, choice_<lv>_v<j>,
reason_<lv>_v<j>. A model that declines yields choice=None and a reason of
"REFUSED: <what it said>"; a request that never parsed yields "FAILED". Both markers
are what utils.REFUSED_REASON_PREFIXES / FAILED_REASON_VALUES match downstream.

The run mechanics (fan-out, checkpointing, --patch, --overwrite_changed) live in
generation.py and are shared with the indirect speeches track.
"""
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from utils import call_api, extract_json
from answer_generation.generation import Track, main

MAX_RETRIES = 5


def load_units(path: str) -> dict[str, list]:
    """{language: [(prompt_template, options), ...]} -- one unit per option ordering."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    prompts, option_lists = data["prompts"], data["option_lists"]
    return {
        language: [(prompts[language], options) for options in option_lists[language]]
        for language in option_lists
        if language in prompts
    }


def call_survey(statement, unit, model):
    template, options = unit
    prompt = template.format(question=statement, options=options)
    last_content = None
    for attempt in range(MAX_RETRIES):
        try:
            response = call_api(prompt, model)
            content = response["choices"][0]["message"]["content"]
            if content is None:
                finish_reason = response["choices"][0].get("finish_reason")
                raise ValueError(
                    f"API returned null content (finish_reason={finish_reason!r}), "
                    f"full message: {response['choices'][0]['message']}")
            last_content = content
            result = extract_json(content)
            if result and "choice" in result and "reason" in result:
                return {"choice": result["choice"], "reason": result["reason"]}
            raise ValueError(
                f"Unexpected response format - could not extract {{choice, reason}} "
                f"from: {content!r}")
        except Exception as error:
            snippet = statement[:80].replace("\n", " ")
            print(f"Error (attempt {attempt + 1}/{MAX_RETRIES}) [{model}] "
                  f"stmt={snippet!r}: {error}", flush=True)
        time.sleep(min(2 ** (attempt + 1), 10))  # exponential backoff
    return {"choice": None,
            "reason": f"REFUSED: {last_content}" if last_content is not None else "FAILED"}


SURVEY_TRACK = Track(
    name="survey (direct Likert)",
    filename="{langs}{variant}.csv",
    primary="choice",
    load_units=load_units,
    call=call_survey,
    # Both framings are keyed by lang_variant inside the negated file, so it serves
    # the base variant too.
    statements_file=lambda variant: "statements_negated.jsonl",
    default_workers=10,
    default_variant="",
    default_prompts="survey_processor_concurrent.json",
    failure_noun="failed/refused",
)


if __name__ == "__main__":
    main(SURVEY_TRACK)
