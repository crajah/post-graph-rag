"""Regrade stored reader-sweep answers under LongMemEval's official protocol.

Zep's published figures were graded by a single GPT-4o using the benchmark's
question-type-specific prompts (Rasmussen et al. 2025, section 4.3: "For answer
evaluation, we employed GPT-4o with the question-specific prompts provided in
[LongMemEval]"). Our own panel is stricter -- three judges, none sharing a
family with the answering model, majority vote. This script scores the same
stored answers the way Zep scored theirs, so the comparison against their
71.2% carries no protocol caveat at all: same generation model, same judge
model, same judge prompts, same verdict rule.

The prompts below are verbatim from the benchmark's evaluate_qa.py, including
the temporal off-by-one allowance and the knowledge-update mixed-answer rule.
The verdict rule is theirs too: 'yes' anywhere in a lowercased reply counts,
temperature 0, max_tokens 10. Abstention instances are those whose question_id
carries '_abs', exactly as upstream.

By default only each arm's first stored repeat is graded -- the official
protocol scores one response per question, and grading one keeps it
like-for-like. --all-repeats grades every stored answer and reports per-repeat.

    python evaluation/longmemeval/regrade_official.py reader_sweep_gpt4.json \
        --arms gpt-4o gemini-3.6-flash
"""
import argparse
import asyncio
import json
import pathlib
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import os                                          # noqa: E402

from openai import AsyncOpenAI                     # noqa: E402

# --- verbatim from LongMemEval src/evaluation/evaluate_qa.py -----------------
PROMPTS = {
    "default": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "abstention": "I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.",
}


def official_prompt(qtype, qid, question, gold, answer):
    if "_abs" in qid:
        return PROMPTS["abstention"].format(question, gold, answer)
    template = PROMPTS.get(qtype, PROMPTS["default"])
    return template.format(question, gold, answer)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="a reader_sweep output file")
    ap.add_argument("--arms", nargs="*", default=None,
                    help="arms to regrade (default: all in the file)")
    ap.add_argument("--judge", default="gpt-4o",
                    help="grading model; gpt-4o is Zep's published protocol")
    ap.add_argument("--data", default=str(HERE / "oracle.json"))
    ap.add_argument("--all-repeats", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = pathlib.Path(args.results)
    if not path.is_absolute():
        path = HERE / path
    sweep = json.loads(path.read_text())
    arms = args.arms or sweep["arms"]
    qtext = {x["question_id"]: x["question"]
             for x in json.loads(pathlib.Path(args.data).read_text())}

    # The raw client, not LLMService: the official protocol pins temperature 0
    # and max_tokens 10, and replicating it exactly matters more than retries.
    client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_API_BASE", "http://localhost:4000/v1"))
    sem = asyncio.Semaphore(args.concurrency)
    verdicts = defaultdict(lambda: defaultdict(list))  # arm -> qid -> [bool]
    rows = []

    async def grade(arm, inst, ridx, answer):
        qid = inst["question_id"]
        prompt = official_prompt(inst["type"], qid, qtext[qid],
                                 inst["gold"], answer)
        text = ""
        async with sem:
            # Official settings: temperature 0, max_tokens 10, n=1; the
            # verdict below is the substring test upstream applies.
            for attempt in range(5):
                try:
                    completion = await client.chat.completions.create(
                        model=args.judge,
                        messages=[{"role": "user", "content": prompt}],
                        n=1, temperature=0, max_tokens=10)
                    text = (completion.choices[0].message.content or "").strip()
                    break
                except Exception as e:
                    if attempt == 4:
                        text = f"GRADER_ERROR: {e}"
                    else:
                        await asyncio.sleep(4 * (attempt + 1))
        label = "yes" in text.lower()
        verdicts[arm][qid].append(label)
        rows.append({"arm": arm, "question_id": qid, "type": inst["type"],
                     "repeat": ridx, "label": label, "judge_reply": text[:40]})

    tasks = []
    for inst in sweep["instances"]:
        for arm in arms:
            runs = inst["arms"][arm]["runs"]
            take = runs if args.all_repeats else runs[:1]
            for ridx, r in enumerate(take):
                tasks.append(grade(arm, inst, ridx, r["answer"]))
    print(f"grading {len(tasks)} answers with {args.judge} "
          f"(official LongMemEval prompts, temp 0)", flush=True)
    await asyncio.gather(*tasks)

    print("\n" + "=" * 66)
    summary = {}
    for arm in arms:
        by_type = defaultdict(list)
        for inst in sweep["instances"]:
            qid = inst["question_id"]
            if verdicts[arm].get(qid):
                # first repeat is the like-for-like verdict
                by_type[inst["type"]].append(1 if verdicts[arm][qid][0] else 0)
        overall = [v for vs in by_type.values() for v in vs]
        print(f"\n  {arm}  (official protocol, {args.judge} judge)")
        for t in sorted(by_type, key=lambda t: -sum(by_type[t]) / len(by_type[t])):
            v = by_type[t]
            print(f"    {t:<28} {sum(v)/len(v):6.1%}  (n={len(v)})")
        print(f"    {'OVERALL':<28} {sum(overall)/len(overall):6.1%}  (n={len(overall)})")
        summary[arm] = {"overall": sum(overall) / len(overall),
                        "by_type": {t: sum(v)/len(v) for t, v in by_type.items()},
                        "n": len(overall)}

    out = pathlib.Path(args.out) if args.out else path.with_name(
        path.stem + f"_official_{args.judge.replace('/', '_')}.json")
    out.write_text(json.dumps({
        "source": path.name, "judge": args.judge,
        "protocol": "LongMemEval evaluate_qa.py verbatim; Zep's published arrangement",
        "summary": summary, "rows": rows}, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
