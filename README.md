# Process Context Graph

Two Python stages that answer one question: **given an email that replaced a
stretch of a process log, can we recover the activities it hid — and does a
context graph help?**

- `ocel_reconstruction/` — *predict*. For every hidden email, produce the
  ordered list of activities that email reports.
- `ocel_evaluation/` — *score*. Compare each prediction against the ground
  truth and emit the metrics.

They are a pipeline: reconstruction writes `predicted.jsonl`, evaluation reads
it. Nothing else is shared between them, and evaluation touches no model, no
network and no credentials.

---

## 1. The task

The upstream corpus is an OCEL 2.0 event log of a loan-origination process. For
each case, a contiguous run of one or more real events was **deleted** and
replaced by a single placeholder event of type `internal_email_sent`, carrying
a generated internal status email that narrates what the employee did.

So the log now reads:

```
- 2016-03-10 | A_Submitted
- 2016-03-12 | internal_email_sent   ← 1..n real activities were deleted here
- 2016-03-18 | O_Accepted
```

The reconstruction task is to name those deleted activities, **in order**,
from the evidence still available. The ground truth is kept in the record
(`true_members`) but is never shown to the model.

Two properties make the task honest:

- **No length hint.** The model is never told how many activities were hidden,
  so over-prediction (hallucinating extra steps) and under-prediction (dropping
  real ones) are both observable and both penalised.
- **No leakage.** Every route by which the answer could reach the model is
  closed deliberately — see [§6 Leakage controls](#6-leakage-controls). This is
  the part of the design that most deserves scrutiny.

### Evidence sources

Three independent sources can be switched on or off, which is how the
experimental arms are formed:

| Source | Flag | What the model sees |
|---|---|---|
| Email | `--use-email` | Subject, body, sender, related objects of the hidden email |
| Observed timeline | `--use-ocel` | The corrupted case timeline, past **and** future, with the hidden email marked `>>> THIS` |
| Context graph | `--use-graph` | The case's subgraph (decisions, rationales, evidence spans, actors) plus, optionally, retrieval tools over the whole graph |

---

## 2. The context graph

The graph is built upstream (not in this repo) by extracting an ontology from
the case's messages and process log. It is consumed here as a single exported
`graph.json` — an APOC JSONL dump, one node or relationship per line.

Node types:

| Node | Meaning |
|---|---|
| `ProcessInstance` | The loan application / case (its OCEL case object) |
| `ActivityCluster` | **Unnamed** placeholder for the work this case ran — one per case |
| `DecisionNode` | A choice point (`decision_type`, `final_choice`, `outcome_status`) |
| `RationaleNode` | The *why* behind a decision |
| `EvidenceNode` | A verbatim span from a message or attachment |
| `Actor` | A person or team |

Key edges:

```
ActivityCluster -has_decision->  DecisionNode
DecisionNode    -justified_by-> RationaleNode
DecisionNode    -supported_by-> EvidenceNode
ActivityCluster -produced->     EvidenceNode
EvidenceNode    -explained_by-> RationaleNode
Actor           -made_decision->DecisionNode
DecisionNode    -acted_on->     ActivityCluster
Actor           -linked_to->    ActivityCluster
ActivityCluster -part_of->      ProcessInstance
```

**The activities are never named in the graph.** The `ActivityCluster` records
only that work happened and what it touched. Naming it is the task. The
decisions, rationales and evidence hanging off the cluster are the *traces* the
hidden activities left behind.

`DecisionNode` and `RationaleNode` additionally carry two vector properties,
written by the upstream ingest:

- `text_embedding` — semantic (Azure OpenAI) embedding of the node text.
- `fastrp_embedding` — structural (Neo4j GDS FastRP) embedding.

These drive the hybrid similarity search. If either is missing on any such
node, `GraphIndex` **raises at construction** and names the offending ids — a
partial ingest fails loudly instead of silently degrading the graph arm.

### How a subgraph is rendered

`src/graph_json.py` is the single place that decides what a subgraph looks like
in a prompt. Nodes are tagged by type (`E`/`D`/`R`/`A`/`AC`/`P`) and rendered as
`TAG Label(details)`, followed by relationship lines:

```
Node:
E1 EvidenceNode("I accepted the application this morning", this email, span=1)
E2 EvidenceNode("offer paperwork is now out with the customer", this email, span=2)
AC1 ActivityCluster(no details)
P1 ProcessInstance(application_1052164642__1gram__occ0000__at0003)
D1 DecisionNode(approve | decision_type=credit | outcome_status=accepted)
R1 RationaleNode("income and collateral both cleared policy")

Relationships:
AC1 is produced E1
AC1 is produced E2
AC1 is has_decision D1
D1 is justified_by R1
AC1 is part_of P1
```

`span=<n>` is reading order within the hidden email's body. The agent is
instructed to walk the spans in ascending order and give each a verdict, which
is what makes the predicted *order* traceable to the email's narrative rather
than to the catalog's listing order or to what "usually happens next".

---

## 3. `ocel_reconstruction`

### Input

A folder containing `generated.jsonl` — one JSON object per hidden email:

| Field | Meaning |
|---|---|
| `case_id` | Corrupted case id, e.g. `Application_1052164642__1g__occ0000__at0003` |
| `graph_key` | The same case as keyed in the graph, e.g. `application_1052164642__1gram__occ0000__at0003` |
| `email` | `{subject, body, sender, objects}` of the hidden email |
| `observed_timeline` | Rendered corrupted case timeline with `>>> THIS` marking the hidden email |
| `true_members` | **Ground truth** — the ordered activities that were hidden |
| `gram_size` / `unit_id` | Size of the hidden run and which pattern it was |
| `original_event_ids` | Ground-truth event ids of the deleted events |

Beside `generated.jsonl`, one `<graph_key>/filled_ocel.json` per case holds that
case's full corrupted OCEL — this is what the `get_ocel_of_application` tool
serves (after sanitising, see §6).

Also required:

- `--process-workdir` — folder holding `schema_<name>_only.json`, the activity
  catalog (`event_type` + `description` per activity). Selected by
  `--schema-name` (default `bank_loan`). The **full** catalog is always used,
  never a restriction to "email-capable" activities: a hidden activity may be
  one that is never itself the subject of an email, and restricting the catalog
  would make it unreconstructable by construction.
- `--graphs` — folder holding the exported `graph.json`. Required when
  `--use-graph true`.

### The two prediction paths

Which path runs is decided solely by `--use-graph`.

#### a) Single-shot (`--use-graph false`) — `src/sequence_predictor.py`

One Azure OpenAI chat call per email, JSON-mode. The prompt is assembled from
`src/prompts.yaml` as: catalog prefix + email block (if `--use-email`) + OCEL
timeline block (if `--use-ocel`) + task block. Each source block carries its own
instructions and no source is framed as authoritative over another.

Returned activity names are normalised case-insensitively against the catalog;
anything not in the catalog is dropped. **Order is preserved and duplicates are
kept** — a real trace can repeat an activity.

Identical prompts are cached in-process (`RECON_CACHE=0` disables), which
matters because the same non-pattern email recurs across experiment levels.

#### b) Agentic / graph (`--use-graph true`) — `src/agent_predictor.py`

An AutoGen `AssistantAgent` per email. Its **cold-start context** (the task
prompt) always contains: the catalog, the case's 2-hop subgraph around its
`ActivityCluster`, the observed timeline (if `--use-ocel`), and seed
`DecisionNode` / `RationaleNode` ids.

`--use-agentic-tools` then splits this arm in two:

- **`true` (default)** — four retrieval tools are bound, and the agent may call
  them for up to `--max-tool-iterations` sequential rounds before answering.
- **`false`** — no tools, no retry; the agent answers single-shot from the
  cold-start context alone. This is the control that measures whether the tool
  loop earns its cost over the graph context by itself.

##### The four tools (`src/tools.py`)

| Tool | Returns |
|---|---|
| `find_similar_decisions(decision_node_id, k, hops)` | Nearest `DecisionNode`s by hybrid score, each with its `hops`-neighbourhood and owning application id |
| `find_similar_rationals(rationale_id, k, hops)` | Same, for `RationaleNode`s |
| `get_ocel_of_application(application_id)` | That application's sanitised corrupted OCEL timeline |
| `get_application_context_subgraph(application_id, hops)` | The `hops`-neighbourhood around that application's `ActivityCluster` |

The hybrid similarity score is

```
score = alpha * cosine(text_embedding) + (1 - alpha) * cosine(fastrp_embedding)
```

with `alpha = --sim-alpha` (default `0.5`), so semantic and structural
neighbourhood are weighted explicitly.

The intended loop: take a seed id from the task → find a comparable case →
read that case's observed log → learn how such a log *names* the steps → then
map this email's spans to catalog activities. A comparable case is the only
place the agent can learn that mapping, so if the agent answers without ever
calling a similarity tool, it is **re-asked once** with the protocol spelled
out (`retried_for_tool_use` in the transcript records this).

##### Traversal rules in `GraphIndex`

- `Actor` nodes are rendered when reached but never traversed **through** —
  actors are merged globally across cases, so expanding through one would fuse
  unrelated applications into a single "island".
- A foreign `ProcessInstance` or `ActivityCluster` is likewise terminal.
- A `ProcessInstance` with no rendered evidence/decision/rationale/cluster
  neighbour is dropped rather than rendered bare.
- `Event` / `Activity` nodes are dropped unless the run explicitly includes
  them (`--filter-event-node false`) — they name the true activity.

### Output

Written to `--out`:

- **`predicted.jsonl`** — every input record echoed verbatim plus two fields:
  `pred` (the ordered predicted activities) and `confidence` (0–1, self-reported
  by the model). This is exactly what `ocel_evaluation` consumes.
- **`<graph_key>/llm_prompts.md`** — one per email. For the single-shot path:
  the system / user / assistant exchange. For the agentic path: system prompt,
  task, the full run transcript including every tool call and result, and the
  final answer. This is the audit trail for any number that ends up in a paper.

### Flags

| Flag | Default | Effect |
|---|---|---|
| `--emails` | *(required)* | Folder with `generated.jsonl` |
| `--out` | *(required)* | Output folder |
| `--process-workdir` | *(required)* | Folder with `schema_<name>_only.json` |
| `--schema-name` | `bank_loan` | Selects the catalog inside the workdir |
| `--schema` | — | Explicit catalog path, overrides the two above |
| `--graphs` | — | Folder with `graph.json`; required with `--use-graph true` |
| `--use-email` | `true` | Show the hidden email's subject/body |
| `--use-ocel` | `true` | Append the observed case timeline |
| `--use-graph` | `true` | Use the graph path (agentic) instead of single-shot |
| `--use-agentic-tools` | `true` | Graph arm only: bind the retrieval tools |
| `--filter-event-node` | `true` | Drop activity-leaking `Event`/`Activity` nodes |
| `--mask-execution-verbs` | `true` | Collapse Actor execution verbs to `linked_to` |
| `--exclude-same-case` | `true` | Refuse retrieval from sibling occurrences of the same case |
| `--sim-alpha` | `0.5` | Semantic vs structural similarity weight |
| `--default-k` | `5` | Hits per similarity call |
| `--default-hops` | `1` | Neighbourhood radius attached to tool results |
| `--max-tool-iterations` | `6` | Max sequential tool rounds per email |
| `--agent-model` | `gpt-5.4` | Model name reported to AutoGen |
| `--reasoning-effort` | `high` | `minimal`/`low`/`medium`/`high`; empty = model default |
| `--email-activity` | `internal_email_sent` | OCEL type of the placeholder event |
| `--workers` | `8` | Parallel predictions (1 = sequential) |
| `--azure-endpoint` / `--azure-deployment` / `--azure-api-version` | — | Pushed into the env the client reads |

Booleans are strict: `true/false/1/0/yes/no/on/off/y/n/t/f`, any case. Anything
else **raises**. This is deliberate — a typo like `"ture"` silently inverting a
run's configuration has happened before, and the only trace was a log line.

### Common arms

```bash
# Email only — the baseline
--use-email true  --use-ocel false --use-graph false

# Timeline only — no email at all
--use-email false --use-ocel true  --use-graph false

# Email + timeline
--use-email true  --use-ocel true  --use-graph false

# Graph, cold start (subgraph + timeline in the prompt, no tools)
--use-email true  --use-ocel true  --use-graph true --use-agentic-tools false

# Graph, full agentic retrieval
--use-email true  --use-ocel true  --use-graph true --use-agentic-tools true
```

---

## 4. `ocel_evaluation`

Reads `predicted.jsonl`, scores `true_members` vs `pred` per email, aggregates.
**Pure standard library** — no model, no network, no credentials, fully
deterministic and reproducible.

### Per-pair metrics (`src/metrics.py`)

Activities are opaque tokens; a single-activity email is just a sequence of
length one.

| Metric | Order-aware? | Meaning |
|---|---|---|
| `edit_distance`, `normalized_edit_distance` | yes | Token-level Levenshtein, normalised by the longer sequence (0 = identical) |
| `multiset_precision/recall/f1` | no | Overlap via per-token minimum counts — partial credit for correct-plus-extra |
| `seq_precision/recall/f1` | yes | Same shape but overlap = LCS length (ROUGE-L) |
| `jaccard` | no | Set overlap `\|A∩B\| / \|A∪B\|` |
| `dice` | no | `2\|A∩B\| / (\|A\|+\|B\|)` |
| `cosine` | no | Cosine of the activity count vectors |
| `lcs_ratio` | yes | `LCS / max(len)` |
| `kendall_tau` | yes | Rank correlation over activities present in both |
| `length_bias` | — | `len(pred) - len(true)`: **positive = over-prediction** |
| `exact_match` | yes | Exact list equality |

Because `LCS ≤ multiset overlap` always, every `seq_*` value is `≤` its
`multiset_*` counterpart — **the gap between them is precisely the ordering
penalty**, which is the pair worth reporting together.

### Error taxonomy

Each pair is labelled by `categorize_pair`:

`exact` · `ordering` (right activities, wrong order) ·
`multiplicity(over|under)` (right activity types, wrong counts) ·
`over-generation` (invented types) · `under-generation` (missed types) ·
`substitution/mixed`.

### Confusion matrix

From a Levenshtein **alignment** of `true` vs `pred`, each aligned token
position is counted as match / substitution / insertion / deletion, with
`__insert__` and `__delete__` as sentinel row/column. Ties prefer substitution
over insert+delete, so `[A]` vs `[B]` reads as `A → B` rather than as two
unrelated errors. This makes systematic confusions (e.g.
`A_Validating → W_Validate application`) visible even when the two sequences
have different lengths.

### Output

Written to `--out`:

- **`summary.json`** — overall roll-up, breakdown by hidden-run size `n`,
  breakdown by dropped unit, and the confusion matrix (overall and per size).
  Every mean carries its population standard deviation alongside.
- **`summary.md`** — the same, as readable tables (`mean ± std`).
- **`scored_rows.jsonl`** — one row per email: `true`, `pred`, confidence,
  category, per-activity false positives / false negatives, alignment ops and
  substitution pairs. This is the file to go to when a headline number needs
  explaining.

Records without an `email` field are skipped as placeholders.

---

## 5. Running it

### Install

Python 3.12.

```bash
pip install openai azure-identity pyyaml numpy \
            autogen-agentchat "autogen-ext[openai,azure]" pytest
```

`ocel_evaluation` needs none of these — it is stdlib only.

### Configure Azure OpenAI

Reconstruction needs a chat deployment. Auth precedence:

1. `AZURE_OPENAI_API_KEY` if set (key-based),
2. otherwise `DefaultAzureCredential` — env vars, managed identity, `az login`.

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="<chat-deployment>"
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
export AZURE_OPENAI_API_KEY="..."        # or rely on DefaultAzureCredential
```

Never pass the key on the command line.

### Run

```bash
python ocel_reconstruction/run.py \
  --emails           /path/to/emails \
  --graphs           /path/to/graph \
  --process-workdir  /path/to/workdir \
  --schema-name      bank_loan \
  --out              /path/to/reconstructed \
  --use-email true --use-ocel true --use-graph true --use-agentic-tools true \
  --workers 8

python ocel_evaluation/run.py \
  --reconstructed /path/to/reconstructed \
  --out           /path/to/evaluation
```

`--help` on either script prints the full flag list.

### Tuning environment variables

| Variable | Default | Effect |
|---|---|---|
| `RECON_MAX_COMPLETION_TOKENS` | `2000` | Completion budget (reasoning models spend part of it on hidden reasoning) |
| `RECON_REASONING_EFFORT` | `high` | Default reasoning budget when the flag is not passed |
| `RECON_RATE_LIMIT_RETRIES` | `6` | 429 retries before giving up |
| `RECON_RATE_LIMIT_BASE_SLEEP` | `5` | Backoff base, seconds |
| `RECON_RATE_LIMIT_MAX_SLEEP` | `60` | Backoff cap, seconds |
| `RECON_CACHE` | `1` | In-process prompt dedup cache; `0` disables |
| `RECON_MIN_GRAPH_COVERAGE` | `0.9` | Minimum share of records that must resolve to a `ProcessInstance` |

### Tests

```bash
python -m pytest -q
```

25 hermetic unit tests covering prompt assembly, subgraph rendering, the
same-case guard and boolean parsing. They make no network calls and need no
credentials.

**What green does and does not mean.** It means the code imports and those
assertions hold. It says nothing about whether reconstruction *accuracy* moved.
The only measurement of accuracy is an `ocel_evaluation` run, and it needs live
model calls and real data. Any accuracy claim must cite `summary.json` and
`scored_rows.jsonl` from an actual run; a change to `prompts.yaml` or to a
system prompt in `agent_predictor.py` alters behaviour while leaving every test
green, so "not measured" is the honest thing to write when no run backs it.

---

## 6. Leakage controls

The ground truth is present in the input files, so every path by which it could
reach the model is closed on purpose. These are the guarantees a reviewer
should check first.

1. **Answer attributes are always stripped.** `true_members` and
   `original_event_ids` are removed from any OCEL rendered to the model,
   unconditionally, on every arm — `src/ocel_lookup.py`.

2. **`Event` / `Activity` nodes are dropped from subgraphs**
   (`--filter-event-node true`). Those nodes name the true activity outright.

3. **Actor execution verbs are masked** (`--mask-execution-verbs true`). The
   six verbs `prepared / reviewed / ran / completed / approved / signed_off`
   pointing at a case's `ActivityCluster` collapse to a neutral `linked_to`.
   Each verb is a model's one-word summary of what an actor did, so it hints at
   the *kind* of activity behind it; masking keeps the actor attached to the
   case without that hint. The system prompt's ontology gloss tracks this flag,
   so the agent is never told about edges it will not see.

4. **Sibling occurrences are refused** (`--exclude-same-case true`). The corpus
   samples a few base applications many times, each occurrence hiding a
   *different* email. A sibling's observed timeline therefore still shows the
   activities this case must predict. Excluding siblings from the similarity
   *ranking* is not enough on its own, because the tools take an application id
   as an argument and the agent can pick one up anywhere a `ProcessInstance` is
   rendered — so `get_ocel_of_application` and
   `get_application_context_subgraph` also refuse a sibling id outright, with
   an explanation the agent can act on. The tool-level guard follows the index
   setting, so a run that deliberately permits same-case retrieval stays
   internally consistent.

5. **The current application is refused too**, for budget rather than
   correctness: its timeline and subgraph are already quoted verbatim in the
   task, so re-fetching either only burns tool iterations. The refusal message
   redirects the agent to the similarity tools.

6. **`--use-email false` redacts, it does not merely omit.** On the graph arm
   the hidden email's `Subject` and `Body` are replaced with `[redacted]` inside
   `get_ocel_of_application`, so a no-email arm cannot recover the email text
   through a tool call.

7. **Graph coverage is enforced.** A graph that resolves none of the records
   still produces a complete, green run — every prompt simply carries "no
   matching `ProcessInstance`" and the agent falls back to the timeline. That is
   a graph-less baseline wearing the graph arm's label. So the run **fails** if
   coverage falls below `RECON_MIN_GRAPH_COVERAGE` (default 90%), naming the
   unresolved ids.

8. **Broken clients abort the run.** Missing deployment, auth failure or
   exhausted quota raise `FatalAzureError` and stop everything, rather than
   producing a file full of empty predictions that would score as a legitimate
   (very bad) result. Transient 429s are distinguished from these and retried
   with backoff; content-level failures (one email's malformed JSON) yield an
   empty prediction for that email alone and are recorded in its
   `llm_prompts.md`.

---

## 7. Layout

```
ocel_reconstruction/
  run.py                    CLI entry point
  src/
    experiment.py           orchestration, parallelism, output writing
    sequence_predictor.py   single-shot path: prompt build, call, normalise
    agent_predictor.py      agentic path: system prompt, AutoGen agent, retry
    tools.py                the four retrieval tools + the refusal guards
    graph_index.py          in-memory graph.json index, similarity, neighbourhoods
    graph_json.py           the one place subgraphs are rendered to text
    ocel_lookup.py          per-application corrupted OCEL, sanitised
    prompt_store.py         loads prompts.yaml, fills {{placeholders}}
    prompts.yaml            externalised single-shot prompts
    azure_client.py         Azure OpenAI client factory + fatal-error detection
  tests/                    hermetic unit tests

ocel_evaluation/
  run.py                    CLI entry point
  src/
    experiment.py           scoring, aggregation, summary.json / summary.md
    metrics.py              all metric definitions (pure stdlib)
```

The agentic path's prompts live in `agent_predictor.py`, not in `prompts.yaml`;
`prompts.yaml` covers the single-shot path only. Both are load-bearing for the
results — treat either as a change that needs a measurement run behind it.

