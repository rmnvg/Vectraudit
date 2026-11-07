# Retrieval poisoning and injection checks

A retrieval-augmented system pastes retrieved documents into a prompt. A document that holds
instructions, role markers or hidden text can steer the model that reads it (indirect prompt
injection), and a document written to rank for a question can crowd out the right answer
(retrieval poisoning). This page covers the checks Vectraudit runs on that risk. They are
local and deterministic: no model, no network, nothing written to your backend, and no
document text in any report.

The attack families follow [promptfoo](https://github.com/promptfoo/promptfoo)'s RAG
red-team plugins (`rag-poisoning`, `indirect-prompt-injection`, `rag-document-exfiltration`;
MIT) and its encoding strategies, reimplemented here without its code.

## Scan a corpus

```bash
vectraudit workbench corpus-scan corpus.jsonl [--patterns extra.json] --out scan.json
```

The corpus is JSONL with `doc_id` (or `id`) and `text` per line. Repeated IDs are allowed,
since chunks of one source document share its `doc_id`; each finding records its row.

| Detector | Severity | Finds |
|---|---|---|
| `instruction_override` | high | "ignore (all) previous instructions", "disregard the above rules" and variants |
| `role_reassignment` | low | "you are now ...". Ordinary prose says it too, so it is a weak signal alone |
| `prompt_reference` | medium | "system prompt", "developer message" |
| `chat_template_token` | high | `<\|im_start\|>`, `[INST]`, `<<SYS>>`, `### System:`, a line starting `assistant:` |
| `zero_width_character` | medium | U+200B to U+200F, U+2060, U+FEFF |
| `bidi_control` | high | U+202A to U+202E, U+2066 to U+2069 |
| `html_comment` | medium | `<!-- ... -->`: text a reader of the rendered page never sees |
| `hidden_style` | high | `display:none`, `font-size:0` |
| `exfiltration_beacon` | high | A Markdown image or link whose URL query holds a template placeholder (`{{x}}`, `${x}`, `{x}`, `<x>`) |
| `encoded_instruction` | high | A long base64 or hex run that decodes to one of the phrases above |
| `mixed_script_word` | low | A word mixing Latin with Cyrillic or Greek letters (a homoglyph candidate). Common unit and maths symbols (Ohm, mu, pi, Delta, Sigma, theta, lambda) are ignored, and so are words under three letters |

Each detector has an ID, a version and a plain-language explanation in the report.

**A finding is a hypothesis.** It marks a document for a person to read. It does not show the
document is unsafe (scraped web text does say "you are now ...", and contains real HTML
comments and science notation with Greek letters), and no findings does not show a corpus is
safe: the scan only finds known traces, and an attacker can avoid every pattern. On the
20,000-chunk web corpus of the RAG demo it flags about 0.1% of chunks, among them real
homoglyph spam such as a word spelled with a Cyrillic letter in place of a Latin one. It
finds nothing in the small corpora under `examples/`.

The report (`vectraudit.corpus_scan`, version 1, evidence level `hypothesis`) holds, per
flagged document: its ID, its row, and each finding's detector, character offsets, severity
and evidence level (at most 5 matches per detector and 20 findings per document, with a count
of the rest). It has no document text. `review` lists the flagged documents to read first,
worst severity and then most findings first. The Checks tab shows the report.

**Excerpts.** `--show-excerpts` prints at most 120 characters around each match **to the
terminal only**. Documents are untrusted and a terminal acts on escape sequences, so control
and formatting characters are printed as `\uXXXX`, never raw. Excerpts never enter the JSON.

**Your own detectors.** `--patterns` takes a JSON list of
`{"id", "pattern", "explanation", "severity"}` (severity `low`, `medium` or `high`; default
`medium`). The ID is 1 to 40 characters of `a-z`, `0-9`, `_`, `.` and `-`, and the finding is
reported as `custom:ID`. A pattern is at most 300 characters, must compile and must not
match the empty string, and at most 50 are allowed. They are regular expressions run locally
on your own corpus: avoid nested repetition.

**Speed.** About 14 seconds for 100,000 documents of 600 characters on a laptop (the budget is
60), because most documents skip the Unicode detectors and long-run decoding.

`--fail-on-findings` exits 3 when any document is flagged.
