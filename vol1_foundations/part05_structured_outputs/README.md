# Part 5 — The JSON Straitjacket

> *Forcing a model to speak structured data without losing its mind (or its schema).*

## What's here

| File | What it does |
|------|--------------|
| [`schema.py`](./schema.py) | Pydantic models (PatientIntake, Invoice) plus the repair code — `extract_json`, `repair_json`, `parse_json_lenient` — that fixes the boring 95% of malformed-JSON breakage. |
| [`structured_client.py`](./structured_client.py) | The ask → parse → validate → retry loop. Embeds the schema, validates against it, and on failure feeds the *specific* validation error back to the model. |
| [`extractors.py`](./extractors.py) | Two cheap tricks — prefill the opening brace, stop-sequence on the closing brace — that get clean JSON on the first try more often. |

## Example

| File | Pattern | Story |
|------|---------|-------|
| [`examples/patient_intake.py`](./examples/patient_intake.py) | Validate-and-retry + domain guardrail | Pulling typed, validated fields out of a messy hospital admissions form — and catching the registration number filed where the age should be |

## Run

```bash
python -m vol1_foundations.part05_structured_outputs.examples.patient_intake
```

## The core idea

The model will get the JSON *almost* right, almost every time. It wraps
it in a markdown fence. It adds "Of course! Here are the details:". It
uses Python's `None` instead of `null`. None of these are deep failures —
they're the model being chatty in ways that break `json.loads()`.

So the playbook, in order of cost:

1. **Prefill `{`** (extractors.py). The cheapest fix. The model can't
   write a preamble if it's already inside the object.
2. **Repair common breakage** (schema.py). Strip fences, fix smart
   quotes, swap Python literals, drop trailing commas. Thirty
   deterministic lines.
3. **Validate against a schema** (structured_client.py). Pydantic gives
   you type coercion and range checks, and a readable error
   when something doesn't fit.
4. **Retry with the error fed back** (structured_client.py). When
   validation fails, tell the model *exactly* what was wrong and ask it
   to fix only that. "Try again" gets the same mistake; "field 'age'
   must be >= 0, you returned -3" gets a fix.

Most calls succeed at step 1 or 2. The retry exists for the long tail —
which is exactly the part that pages you at 2 a.m. if you ignore it.

## Why hand the model the schema, not a prose description

`structured_client.py` builds its instructions from
`model_cls.model_json_schema()` — the actual JSON schema, not a paragraph
describing the fields. Two reasons:

- The model follows a schema better than prose.
- There's zero gap between what you asked for and what you validate
  against. They're literally the same object.

## The check the schema can't do

`patient_intake.py` ends with the move that separates people who've been
burned from people who are about to be. After the JSON parses and the
schema validates, it runs one more check the schema *can't* express:
**does the age even make sense for an adult ward (18–110)?**

This catches the dangerous failure — not malformed JSON, which your
parser already catches, but *well-formed JSON full of plausible nonsense.*
The model reads a form where the patient is 54 and the registration number
is 38, and confidently files the age as 38. Every bracket correct, every
type valid, nothing thrown. The record is clean and quietly, dangerously
wrong.

Note where the rule lives: the **schema** (`PatientIntake`) allows ages
0–120 so paediatric and maternity records validate. The narrow adult-ward
rule (18–110) lives in the **example**, not the shared model, because
different wards have different plausible ranges. Don't bake one ward's
rule into a model everyone shares.

## Known limits

- `extract_with_stop_sequence` only works for flat objects — a nested
  object's first `}` would cut the response off early. For nested
  schemas, use `extract_structured` (validate-and-retry) or a real
  tool/function-calling interface.
- The repair code is intentionally not a full JSON-fixer. Anything
  weirder than the common cases should fail loudly and go back through
  the retry loop, not get silently mangled.

## Where this connects

- The clinical sanity check at the end of the patient-intake example
  (is this age plausible?) is a domain guardrail — see
  [`vol5_evals/part25_guardrails/`(Coming)](../../vol5_evals/part25_guardrails/).
- Structured outputs are the foundation for tool calling
  ([`vol3_agents/part13_react_patterns/`(Coming)](../../vol3_agents/part13_react_patterns/)):
  a tool call is just a structured output the model commits to.
