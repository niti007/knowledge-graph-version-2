"""Custom guardrail actions and the pure decision logic behind them.

Everything a rail *decides* lives here as a plain function over plain data --
`assess_grounding`, `apply_citation_policy`, `PiiScanner.scan`,
`looks_like_refusal`. The `@action`-shaped wrappers at the bottom are thin: they
call those functions, write a `RailDecision` into the run ledger, and hand
Colang a boolean. That split is deliberate. A rail whose logic is only reachable
through a Colang interpreter and a live LLM cannot be tested for the failure
mode that actually matters -- firing on a benign lookalike -- so the logic is
tested directly and the wrappers are tested for wiring.

Two design calls are load-bearing and are argued in place below:

1. **The grounding rail does not use the re-rank score alone.** Phase 3 measured
   that a cross-encoder scores topical relevance, not answerability. See
   `assess_grounding`.

2. **PII masking deliberately ignores PERSON and LOCATION.** This corpus is a
   staff directory and a systems inventory; "who leads Infrastructure?" is a
   question whose correct answer *is* a person's name. See `PII_ENTITIES`.
"""

from __future__ import annotations

import contextvars
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ telemetry

@dataclass
class RailDecision:
    """One rail's verdict on one run. This is the Phase 9 scorecard's raw input."""

    rail: str
    stage: str                  # "input" | "output"
    triggered: bool             # did this rail fire?
    blocking: bool              # would firing stop the response reaching the user?
    reason: str = ""            # machine-readable slug, e.g. "unsupported_positive_claim"
    detail: dict = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunLedger:
    """Per-run scratch space shared between the runner and the actions.

    NeMo does expose a `$passthrough_output` context variable, but routing our
    evidence through the Colang context would make the grounding rail depend on
    an internal of the framework AND make it untestable without one. The ledger
    is ours: `runner.py` installs it, the generation step fills in the agent
    response, the actions read it and append decisions.
    """

    question: str = ""
    masked_question: str | None = None
    agent_response: Any | None = None       # AgentResponse, once generation ran
    agent_latency_ms: float = 0.0
    decisions: list[RailDecision] = field(default_factory=list)
    pii_input: list[dict] = field(default_factory=list)
    pii_output: list[dict] = field(default_factory=list)
    citation_decision: "CitationDecision | None" = None
    grounding: "GroundingVerdict | None" = None

    def record(self, decision: RailDecision) -> RailDecision:
        self.decisions.append(decision)
        return decision

    @property
    def fired(self) -> list[str]:
        return [d.rail for d in self.decisions if d.triggered]

    @property
    def blocked_by(self) -> str | None:
        for d in self.decisions:
            if d.triggered and d.blocking:
                return d.rail
        return None


_LEDGER: contextvars.ContextVar[RunLedger | None] = contextvars.ContextVar(
    "guardrails_ledger", default=None)


def current_ledger() -> RunLedger:
    """The ledger for the run in flight; an orphan one if called outside a run."""
    led = _LEDGER.get()
    if led is None:
        led = RunLedger()
        _LEDGER.set(led)
    return led


def set_ledger(ledger: RunLedger | None):
    return _LEDGER.set(ledger)


def reset_ledger(token) -> None:
    _LEDGER.reset(token)


# ----------------------------------------------------------------------- PII

# ---------------------------------------------------------------- PII scope
#
# SCOPE STATEMENT (documented risk acceptance, not an implementation detail).
#
# In scope: contact and identity credentials -- email, phone, SSN, card, IBAN,
# IP, crypto address. These are PII regardless of which corpus they appear in,
# and no legitimate question about ACME's systems needs them echoed back.
#
# Explicitly OUT of scope: PERSON and LOCATION.
#   Rationale. `users.csv` is 46 employees across 6 teams, and the system's
#   headline capability is "which team owns the system Payment-Service depends
#   on, and who leads it?" -> "Infrastructure / Marcus Lee". A PERSON rail masks
#   the answer to the question the corpus exists to answer. The result would
#   look safe in a scorecard while being useless in production -- the worst
#   trade a guardrail can make.
#   Compensating controls, which is why this is acceptance rather than a hole:
#     (1) access control on the corpus itself -- the knowledge base is internal,
#         and every reader of an answer is already authorised to read the
#         directory the answer came from;
#     (2) contact-credential masking (this rail) still strips the fields that
#         would let someone act on a name off-platform -- email, phone, ID;
#     (3) self_check_output independently blocks answers that disclose private
#         personal details (home address, salary, medical, disciplinary), which
#         is the disclosure risk a PERSON rail is actually reaching for.
#   Residual risk accepted: an employee's NAME and their team/role can appear in
#   an answer. That is the product working.
#
# Also out of scope, and these three were REMOVED after validation:
#   US_DRIVER_LICENSE -- a loose alphanumeric pattern. It matched the 8-hex
#     chunk id inside our own citation markers and shipped
#     "[POL-002#<US_DRIVER_LICENSE>]" to a user: a citation that resolves to
#     nothing, which is worse than no citation at all. It hit 4 of 66 chunk-id
#     markers. Every false positive found in validation traced to this
#     recognizer rather than to the score threshold, so the threshold stays.
#   US_PASSPORT -- same family: a bare 9-digit pattern with no structure.
#   MEDICAL_LICENSE -- fired at 1.00 on a substring of an IBAN, and this corpus
#     has no medical content for it to protect.
# Dropping them costs nothing: ACME's corpus contains no driver licences,
# passports or medical licences, so those recognizers had no true positives to
# find and only false ones to produce.
PII_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IBAN_CODE",
    "IP_ADDRESS",
    "CRYPTO",
)

# Calibrated against this corpus rather than picked. Presidio scores a phone
# number 0.75 when the surrounding words say "phone" and 0.4 when they do not
# ("page the DBA at (415) 555-0142" -- exactly the shape an SOP uses), so a 0.6
# threshold silently misses the leak this rail exists to catch.
#
# KNOWN LIMIT, stated rather than smoothed over: an un-cued phone number scores
# EXACTLY 0.40, which is the shipped threshold. The false-negative margin is
# zero. A Presidio release that nudges that recognizer's base score down by any
# amount switches this rail off for un-cued numbers with no test failing
# anywhere -- which is why presidio-analyzer is pinned to an exact version in
# pyproject.toml rather than floored, and why the pin carries this reason.
# `test_an_uncued_phone_number_sits_exactly_on_the_threshold` fails loudly if a
# future version moves it.
PII_SCORE_THRESHOLD = 0.4


@dataclass
class PiiFinding:
    entity_type: str
    start: int
    end: int
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PiiScan:
    text: str
    masked_text: str
    findings: list[PiiFinding] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.findings)

    @property
    def entity_types(self) -> list[str]:
        return sorted({f.entity_type for f in self.findings})


# Citation markers are OURS. `build_citations` in the agent generates them from
# the provenance ledger; they are neither user input nor model prose, so they
# are not a channel through which PII can arrive and scanning them can only ever
# produce a false positive. It did: an 8-hex chunk id scored US_DRIVER_LICENSE
# 0.65 and a user was handed "[POL-002#<US_DRIVER_LICENSE>]" -- a citation
# pointing at nothing. Dropping that recognizer fixes the instance; excluding
# the markers fixes the class. Both are done, because the next loose recognizer
# to be added would otherwise reintroduce it.
_CITATION_MARKER_RE = re.compile(
    r"\[(?:[A-Za-z0-9_.-]+#[0-9a-fA-F]{4,}|graph:[^\]]+)\]")


def _citation_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _CITATION_MARKER_RE.finditer(text)]


def _overlaps_citation(text: str, start: int, end: int) -> bool:
    return any(start < c_end and end > c_start
               for c_start, c_end in _citation_spans(text))


def _internal_email_recognizer():
    """Detect `name@host.internal` and friends, which Presidio's own cannot.

    Presidio's EMAIL_ADDRESS recognizer validates the TLD against the public
    suffix list, so `marcus.lee@acme.internal` produces NOTHING -- not a low
    score, no candidate at all, at any threshold, while `marcus.lee@acme.com`
    scores 1.00. Every one of the 46 employee addresses in `users.csv` uses the
    internal domain, so the PII rail could not fire on the only real PII this
    corpus contains: scanning all 25 documents at threshold 0.01 produced zero
    EMAIL_ADDRESS candidates. "List every employee email" was stopped by
    `self_check_input` alone, with nothing behind it -- the two-pass design was
    claiming a defence in depth that did not exist.

    Scored 0.9 rather than 1.0 because this is a local pattern with no public
    registry to confirm the domain against.
    """
    from presidio_analyzer import Pattern, PatternRecognizer

    pattern = Pattern(
        name="internal_email",
        # The same local-part grammar as a public address, but with the TLDs
        # the public suffix list deliberately omits.
        regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\."
              r"(?:internal|local|corp|lan|intranet|localdomain|home|test|"
              r"example|invalid)\b",
        score=0.9,
    )
    return PatternRecognizer(supported_entity="EMAIL_ADDRESS",
                             name="InternalEmailRecognizer",
                             patterns=[pattern])


class PiiScanner:
    """Presidio analyze + anonymize, wrapped so the rest of the app sees a dataclass.

    The spaCy pipeline is loaded lazily and once: building an AnalyzerEngine is
    ~2s, and a rail that costs two seconds on every request is a rail someone
    turns off.
    """

    def __init__(self, entities: Iterable[str] = PII_ENTITIES,
                 score_threshold: float = PII_SCORE_THRESHOLD,
                 spacy_model: str = "en_core_web_sm"):
        self.entities = list(entities)
        self.score_threshold = score_threshold
        self.spacy_model = spacy_model
        self._analyzer = None
        self._anonymizer = None

    # -- engines
    def _build(self):
        if self._analyzer is not None:
            return
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": self.spacy_model}],
        })
        analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(),
                                  supported_languages=["en"])
        analyzer.registry.add_recognizer(_internal_email_recognizer())
        self._analyzer = analyzer
        self._anonymizer = AnonymizerEngine()

    def scan(self, text: str) -> PiiScan:
        """Find and mask PII. Returns the original text unchanged when clean."""
        if not text or not text.strip():
            return PiiScan(text=text or "", masked_text=text or "")
        self._build()
        results = self._analyzer.analyze(
            text=text, language="en", entities=self.entities,
            score_threshold=self.score_threshold)
        results = [r for r in results
                   if not _overlaps_citation(text, r.start, r.end)]
        findings = [PiiFinding(r.entity_type, r.start, r.end, float(r.score))
                    for r in results]
        if not findings:
            return PiiScan(text=text, masked_text=text)
        masked = self._anonymizer.anonymize(text=text, analyzer_results=results).text
        return PiiScan(text=text, masked_text=masked, findings=findings)


# ------------------------------------------------------- refusal recognition

# A deliberately narrow set. The grounding rail's job is to catch an answer that
# *asserts* something the corpus cannot support; an answer that already says "I
# don't know" is the outcome the rail wants, so firing on it would be a false
# positive that turns a correct abstention into a block. Matching is on
# normalized text and requires one of these explicit phrasings -- a hedge like
# "it appears that" is not a refusal and is correctly left to the rail.
_REFUSAL_PATTERNS = (
    r"\bi (?:do not|don't) (?:have|know|see)\b",
    # Contractions matter more than they look: the agent's most common
    # abstention phrasing in the Phase 5 evidence run was "I couldn't find
    # specific details about ...", and without `couldn'?t` that answer sailed
    # past the citation rail still wearing five citations -- the exact Phase 4
    # defect this rail exists to fix.
    r"\bi (?:cannot|can'?t|could(?: ?n'?t| not)|am unable to|"
    r"was(?: ?n'?t| not) able to) "
    r"(?:find|answer|determine|help|provide|verify|locate|see)\b",
    r"\bi (?:did(?: ?n'?t| not)|could(?: ?n'?t| not)) find\b",
    r"\bno (?:information|results|record|records|data|documents?|evidence|such)\b",
    r"\bnot (?:enough|sufficient) (?:information|context|evidence)\b",
    r"\b(?:there are|there is) no\b",
    r"\bnothing (?:in|found)\b",
    r"\bdoes not (?:appear|contain|include|have)\b",
    r"\bthe corpus (?:does not|doesn't)\b",
    r"\bunable to (?:answer|determine|verify|find)\b",
    r"\bi'?m sorry,? (?:i|but)\b",
    r"\bcould(?: ?n'?t| not) (?:identify|find|locate)\b",
    r"\bno (?:systems?|teams?|people|documents?) (?:depend|are|were)\b",
)
_REFUSAL_RE = re.compile("|".join(_REFUSAL_PATTERNS), re.IGNORECASE)


# A refusal is a statement about the answer AS A WHOLE, and it announces itself
# up front. Matching anywhere produced a real false positive on the system's
# headline multi-hop question: "Auth-DB is owned by Infrastructure, led by
# Marcus Lee ... There is no ownership information available for
# Notification-Service" is a correct, well-supported answer with one honest
# partial negative in its tail, and matching that tail suppressed all three of
# its citations. So a refusal must LEAD.
#
# The first cut of that rule short-circuited to whole-text matching for answers
# under 200 characters, and validation showed the boundary doing real damage:
# the canonical example above is 187 characters, so the very answer the design
# exists to protect fell through the short-circuit and lost all three
# citations. Fourteen more characters and it kept them. A rule whose behaviour
# flips on answer length is not a rule about refusals, and a suite that pads its
# fixture past the boundary (as this one did) never sees it.
#
# So length is gone. What replaces it: find the first SUBSTANTIVE sentence and
# match only there. A leading fragment ("Auth-DB?", "Short answer:") is not a
# thesis, so it is skipped rather than mistaken for the lead -- otherwise
# "Auth-DB? There are no such records." would read as a positive answer.
_FRAGMENT_CHARS = 30


def _lead_sentence(text: str) -> str:
    """The first sentence long enough to be making a claim."""
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    if not parts:
        return text.strip()
    for part in parts:
        if len(part.strip()) >= _FRAGMENT_CHARS:
            return part
    # Every sentence is a fragment: the whole thing is the lead.
    return " ".join(parts)


def looks_like_refusal(text: str) -> bool:
    """True when the answer, AS A WHOLE, declines or reports an absence.

    A heuristic, and labelled as one. It is used only to *suppress* the
    grounding rail and to demote citations, never to block, so a miss costs a
    stricter check and a false hit costs a demoted citation -- neither silently
    lets an unsupported claim through.
    """
    if not text or not text.strip():
        return True
    return bool(_REFUSAL_RE.search(_lead_sentence(text)))


def acknowledges_absence(text: str) -> bool:
    """True when the answer reports an absence ANYWHERE, not only as its thesis.

    Deliberately a different predicate from `looks_like_refusal`, and the two
    rails deliberately use different ones. The distinction was forced by a real
    case in the Phase 5 evidence run:

        "Auth-DB is owned by Infrastructure, led by Marcus Lee. API-Gateway is
         owned by Security, led by Natalia Voss. There is no ownership
         information available for Notification-Service."

    Retrieval's `entity_resolved_but_graph_empty` is True here -- one of the
    three graph lookups came back empty -- so the grounding rail must decide
    whether the answer is claiming something the graph does not hold. It is not:
    it states the gap explicitly. A *partial* acknowledgement is enough to make
    a mixed answer faithful. It is NOT enough to strip the answer's citations,
    because two of its three claims really are cited. Hence:

        grounding rail  -> acknowledges_absence (anywhere)
        citation rail   -> looks_like_refusal   (the answer is wholly a refusal)

    Collapsing these into one predicate fails one way or the other: match
    anywhere and a good multi-hop answer loses its citations; match only the
    lead and that same answer gets blocked outright as ungrounded.
    """
    if not text or not text.strip():
        return True
    return bool(_REFUSAL_RE.search(text))


# ------------------------------------------------------------- grounding rail

# Phase 3, measured on this corpus:
#   - "What depends on DataWarehouse?"  scores 0.9944 when the answer is "nothing"
#   - a non-existent "INC-206"          scores 0.9863
#   - best achievable accuracy on the score alone is 90%, at no threshold
#   - legitimate queries score as low as 0.12 against an off-topic ceiling of
#     0.072 -- a usable margin of ~0.05
#
# The conclusion is not "pick a better threshold", it is "the score answers a
# different question". A cross-encoder ranks passages by topical relevance; it
# has no opinion on whether the corpus contains the fact being asked for. So the
# score is used ONLY where it is measurably reliable -- the bottom of its range,
# where it is a zero-false-negative off-topic detector -- and the near-miss
# cases it provably cannot see are handed to two structural signals that know
# the answer directly:
#
#   entity_resolved_but_graph_empty  the entity is real, the graph holds no such
#                                    fact -> the only faithful answer is negative
#   negative_facts                   the retriever explicitly recorded "none"
#
# Those two are facts about the retrieval, not estimates, which is why they can
# carry the near-miss cases a similarity score cannot.
OFF_TOPIC_RERANK_MAX = 0.10


@dataclass
class GroundingVerdict:
    grounded: bool
    reason: str                     # slug: "" when grounded
    signals: dict = field(default_factory=dict)
    answer_declines: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def assess_grounding(abstention: dict | None, answer: str,
                     off_topic_max: float = OFF_TOPIC_RERANK_MAX) -> GroundingVerdict:
    """Decide whether `answer` is supportable by what retrieval actually found.

    Returns grounded=False only when the evidence says the corpus cannot support
    a positive claim AND the answer makes one anyway. An answer that already
    declines is grounded by definition -- it is asserting the absence the
    evidence reports.
    """
    abstention = abstention or {}
    declines = looks_like_refusal(answer)          # the whole answer is a refusal
    acknowledges = acknowledges_absence(answer)    # the answer states a gap

    score = abstention.get("max_rerank_score")
    signals = {
        "max_rerank_score": score,
        "off_topic_threshold": off_topic_max,
        "any_tool_called": bool(abstention.get("any_tool_called")),
        "entity_resolved_but_graph_empty": bool(
            abstention.get("entity_resolved_but_graph_empty")),
        "negative_facts": list(abstention.get("negative_facts") or []),
        "no_results_returned": bool(abstention.get("no_results_returned")),
        "entity_unresolved": bool(abstention.get("entity_unresolved")),
        "answer_declines": declines,
        "answer_acknowledges_absence": acknowledges,
    }

    def verdict(grounded: bool, reason: str) -> GroundingVerdict:
        return GroundingVerdict(grounded=grounded, reason=reason, signals=signals,
                                answer_declines=declines)

    # (a) OFF-TOPIC. The one place the re-rank score is trustworthy: nothing in
    # the corpus is even about this. Fires regardless of how the answer is
    # phrased, because a confident off-corpus answer and a hedged one are both
    # unsupported.
    if score is not None and score < off_topic_max:
        if declines:
            return verdict(True, "off_topic_but_answer_declines")
        return verdict(False, "off_topic_low_rerank")

    # Everything below is a NEAR MISS: the corpus is about this subject, but the
    # specific fact asked for is not in it. The score cannot see this at all
    # (0.9944 on exactly such a query), so the structural signals decide -- and
    # they only matter if the answer asserted something.
    if acknowledges:
        # The answer already reports the absence the evidence describes, in
        # whole or in part. There is no unsupported positive claim to catch.
        return verdict(True, "answer_declines" if declines
                       else "answer_acknowledges_absence")

    if signals["entity_resolved_but_graph_empty"]:
        return verdict(False, "entity_resolved_but_graph_empty")

    if signals["negative_facts"]:
        return verdict(False, "negative_facts_present")

    if signals["no_results_returned"]:
        return verdict(False, "no_results_returned")

    if not signals["any_tool_called"]:
        # Answered without consulting the corpus at all. That is parametric
        # knowledge wearing an enterprise assistant's voice.
        return verdict(False, "answered_without_retrieval")

    return verdict(True, "")


# -------------------------------------------------------------- citation rail

@dataclass
class CitationDecision:
    citations: list[dict]
    provenance: list[dict]
    suppressed: bool
    reason: str = ""

    def to_dict(self) -> dict:
        return {"citations": self.citations, "provenance": self.provenance,
                "suppressed": self.suppressed, "reason": self.reason}


def apply_citation_policy(answer: str, citations: list[dict] | None,
                          grounded: bool = True) -> CitationDecision:
    """Stop a refusal from carrying citations that look like support.

    Carried from Phase 4. The ledger honestly records what retrieval returned,
    so a declined answer still arrives with a citation list attached. That is
    accurate as *provenance* and misleading as *support*: a user reading "I have
    no record of INC-206" under three document citations reasonably concludes
    those documents were consulted and agreed.

    The fix is not to delete the record -- provenance is the audit trail and
    deleting it would be the dishonest option. Citations move to `provenance`,
    each tagged `supports_answer: False`, and the user-facing citation list goes
    empty. Nothing is lost; the claim of support is.
    """
    citations = list(citations or [])
    declines = looks_like_refusal(answer)
    if not citations:
        return CitationDecision([], [], False, "")
    if declines or not grounded:
        reason = "answer_declines" if declines else "answer_not_grounded"
        demoted = [{**c, "supports_answer": False, "demoted_because": reason}
                   for c in citations]
        return CitationDecision([], demoted, True, reason)
    return CitationDecision(
        [{**c, "supports_answer": True} for c in citations], [], False, "")


# ----------------------------------------------------- action registration

def build_actions(scanner: PiiScanner | None = None) -> dict[str, Callable]:
    """The custom actions, bound to one PiiScanner. `runner.py` registers these.

    Each returns a plain value to Colang and writes a RailDecision to the
    ledger. They are async because NeMo awaits every action.
    """
    scanner = scanner or PiiScanner()

    async def mask_pii_input(text: str = "") -> str:
        t0 = time.perf_counter()
        led = current_ledger()
        scan = scanner.scan(text or led.question)
        led.pii_input = [f.to_dict() for f in scan.findings]
        led.masked_question = scan.masked_text
        led.record(RailDecision(
            rail="pii_input", stage="input", triggered=scan.found, blocking=False,
            reason="pii_masked" if scan.found else "",
            detail={"entity_types": scan.entity_types, "n": len(scan.findings)},
            latency_ms=(time.perf_counter() - t0) * 1000))
        return scan.masked_text

    async def mask_pii_output(text: str = "") -> str:
        """Second pass. Retrieval can pull a phone number out of an SOP and the
        model will repeat it; the input rail never saw that text."""
        t0 = time.perf_counter()
        led = current_ledger()
        scan = scanner.scan(text)
        led.pii_output = [f.to_dict() for f in scan.findings]
        led.record(RailDecision(
            rail="pii_output", stage="output", triggered=scan.found, blocking=False,
            reason="pii_masked_in_answer" if scan.found else "",
            detail={"entity_types": scan.entity_types, "n": len(scan.findings)},
            latency_ms=(time.perf_counter() - t0) * 1000))
        return scan.masked_text

    async def check_grounding(text: str = "") -> bool:
        """True when the answer is grounded. Colang blocks on False."""
        t0 = time.perf_counter()
        led = current_ledger()
        agent = led.agent_response
        abstention = getattr(agent, "abstention", None) if agent is not None else None
        answer = text or (getattr(agent, "answer", "") if agent is not None else "")
        verdict = assess_grounding(abstention, answer)
        led.grounding = verdict
        led.record(RailDecision(
            rail="check_grounding", stage="output", triggered=not verdict.grounded,
            blocking=True, reason=verdict.reason, detail=verdict.signals,
            latency_ms=(time.perf_counter() - t0) * 1000))
        return verdict.grounded

    async def check_citations(text: str = "") -> bool:
        """Never blocks. Rewrites the citation list and records why."""
        t0 = time.perf_counter()
        led = current_ledger()
        agent = led.agent_response
        citations = list(getattr(agent, "citations", []) or []) if agent else []
        answer = text or (getattr(agent, "answer", "") if agent else "")
        grounded = led.grounding.grounded if led.grounding else True
        decision = apply_citation_policy(answer, citations, grounded)
        led.citation_decision = decision
        led.record(RailDecision(
            rail="check_citations", stage="output", triggered=decision.suppressed,
            blocking=False, reason=decision.reason,
            detail={"n_before": len(citations), "n_after": len(decision.citations)},
            latency_ms=(time.perf_counter() - t0) * 1000))
        return True

    # -- self-check wrappers -------------------------------------------
    #
    # These call NeMo's own `self_check_input` / `self_check_output`
    # implementations, but under different action NAMES. That is not
    # decoration, it is required: Colang 1.0 starts any flow whose head matches
    # the current event, so a flow of ours that ran `execute self_check_input`
    # also woke NeMo's built-in `self check input` flow, which then evaluated
    # `$response.is_blocked` against a variable it had never set and crashed the
    # run. Wrapping under our own names keeps the library's prompt handling and
    # variant resolution while giving the built-in flows nothing to match on.
    #
    # They also return a plain bool. Colang's expression evaluator cannot read
    # attributes off a frozen slots dataclass (`RailOutcome`), so handing the
    # object to `if $result.is_blocked` fails at runtime for the same reason.

    async def _self_check(kind: str, rail: str, reason: str,
                          variant: str | None, kwargs: dict) -> bool:
        t0 = time.perf_counter()
        if kind == "input":
            from nemoguardrails.library.self_check.input_check.actions import (
                self_check_input as impl)
            call = dict(kwargs)
            if variant:
                call["variant"] = variant
        else:
            from nemoguardrails.library.self_check.output_check.actions import (
                self_check_output as impl)
            call = dict(kwargs)
        outcome = await impl(**call)
        blocked = bool(getattr(outcome, "is_blocked", False))
        current_ledger().record(RailDecision(
            rail=rail, stage=kind, triggered=blocked, blocking=True,
            reason=reason if blocked else "",
            detail={"decision": str(getattr(outcome, "decision", ""))},
            latency_ms=(time.perf_counter() - t0) * 1000))
        return blocked

    async def check_jailbreak(llms=None, llm_task_manager=None, context=None,
                              events=None, llm=None, config=None) -> bool:
        """True => block. Jailbreak / prompt-injection / harmful request."""
        return await _self_check("input", "self_check_input",
                                 "jailbreak_or_injection", None,
                                 dict(llms=llms, llm_task_manager=llm_task_manager,
                                      context=context, events=events, llm=llm,
                                      config=config))

    async def check_topic_scope(llms=None, llm_task_manager=None, context=None,
                                events=None, llm=None, config=None) -> bool:
        """True => block. Question is not about ACME's corpus at all."""
        return await _self_check("input", "topic_scope", "off_corpus_topic",
                                 "topic_scope",
                                 dict(llms=llms, llm_task_manager=llm_task_manager,
                                      context=context, events=events, llm=llm,
                                      config=config))

    async def check_output_policy(llms=None, llm_task_manager=None, context=None,
                                  events=None, llm=None, config=None) -> bool:
        """True => block. Toxicity, secret disclosure, policy violation."""
        return await _self_check("output", "self_check_output",
                                 "output_policy_violation", None,
                                 dict(llms=llms, llm_task_manager=llm_task_manager,
                                      context=context, events=events, llm=llm,
                                      config=config))

    return {
        "mask_pii_input": mask_pii_input,
        "mask_pii_output": mask_pii_output,
        "check_grounding": check_grounding,
        "check_citations": check_citations,
        "check_jailbreak": check_jailbreak,
        "check_topic_scope": check_topic_scope,
        "check_output_policy": check_output_policy,
    }
