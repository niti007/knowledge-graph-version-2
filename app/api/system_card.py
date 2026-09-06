"""Static answers for questions about the assistant itself.

**Why this lives in the API layer and not in a rail.**

Phase 5 measured a 3/32 benign false-positive rate, and all three were the same
shape: "what can you do?", "which documents do you have access to?", "can you
cite your sources?". The chain was consistent -- `topic_scope` correctly ALLOWED
them (the topic prompt names them explicitly), the agent answered from its own
parametric knowledge without calling a tool, and `check_grounding` then blocked
the answer as `answered_without_retrieval`.

That block is correct behaviour for the rail. `answered_without_retrieval` is
the rule that stops the agent passing off model knowledge as ACME fact, and it
is the single most valuable grounding signal in the system. Teaching it to
recognise "this question is about the assistant, not about ACME" would put
routing logic inside a safety check, one broad pattern away from excusing the
exact class of answer it exists to catch -- "what do you know about our
password policy?" is a corpus question wearing the same words.

So the question never reaches the rails. A meta question is answered from the
card below, before retrieval, and therefore never presents itself as an
unsupported corpus claim. There is nothing to block because nothing was
claimed about the corpus.

**The classifier is deliberately biased towards precision.**

Two rules, both narrow:

1. The question must match an anchored pattern about *the assistant* -- its
   capabilities, its sources, its limits. Free-floating keywords are not
   enough; "sources" alone matches nothing.
2. Even then, any corpus vocabulary in the question (a policy, an SOP, an
   incident id, a system, a person, a security topic) vetoes the match.

A missed meta question costs nothing -- it takes the normal path and is
answered, or honestly refused, like anything else. A *false* match is expensive:
it would answer a real corpus question with a canned paragraph and no citations.
Recall is therefore traded away on purpose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------- the card

SYSTEM_CARD = """\
I'm the ACME Enterprise Knowledge Assistant. I answer questions about ACME's \
internal documentation, and every answer I give is grounded in that corpus and \
returned with citations to the specific documents it came from.

**What I cover.** A 25-document corpus plus structured company data:
- 4 corporate policies (POL-001 to POL-004) - information security, data retention, access control, acceptable use
- 7 standard operating procedures (SOP-01 to SOP-07) - incident response, backup, failover, deployment approval
- 5 incident reports (INC-201 to INC-205) with root causes and resolutions
- 5 technical manuals covering ACME's systems and their dependencies
- an FAQ and supporting reference documents
- people, product and transaction data (46 people across 6 teams, and the \
systems they own), queried as a knowledge graph so I can follow relationships \
such as "which team owns the system Payment-Service depends on, and who leads it".

**How I answer.** I search the document corpus and the knowledge graph, re-rank \
what comes back, and answer only from the retrieved material. Citations name the \
document and section each claim came from - document citations carry a document \
id, graph citations name the query template and the path through the graph.

**What I can't do.** I can't answer from anything outside that corpus, so \
questions about topics ACME hasn't documented get an honest "I don't have \
support for that" rather than a guess. I don't take actions - no changes to \
systems, tickets or records. I don't disclose credentials, keys or personal \
contact details, and personal data in your question is masked before it reaches \
retrieval. I'm not a substitute for the on-call rota or a security incident \
process: for anything live, follow the escalation path in SOP-01.\
"""

# ---------------------------------------------------------- the classifier

# Anchored on the assistant as the subject. Each pattern names both a
# capability verb and the second person; neither alone qualifies.
_META_PATTERNS: tuple[tuple[str, str], ...] = (
    ("capabilities", r"\bwhat (kinds?|sorts?|types?) of (questions?|things?) (can|could) (you|this|it)\b"),
    ("capabilities", r"\bwhat (can|could) (you|this assistant|this bot|this tool) (do|help( me)? with|answer|tell me)\b"),
    ("capabilities", r"\bwhat (do|does) (you|this assistant|this bot|this tool) do\b"),
    ("capabilities", r"\bwhat (are|is) (you|your) (capabilities|able to do|good (at|for)|for)\b"),
    ("capabilities", r"\b(how|what) (can|could|do) (you|i) (help|use you)\b(?!.*\b(with the|with our|with my)\b)"),
    ("identity",     r"^\s*(who|what) are you\b"),
    ("identity",     r"\bhow do you work\b"),
    ("sources",      r"\b(which|what) (documents?|docs?|sources?|files?|data|corpus|knowledge base) (do|can) you (have|use|access|search|cover|read)\b"),
    ("sources",      r"\b(which|what) (documents?|docs?|sources?|data) (do you have )?access to\b"),
    ("sources",      r"\bdo you have access to (any |the )?(documents?|sources?|data)\b"),
    ("sources",      r"\bwhere do (your|the) answers come from\b"),
    ("sources",      r"\bwhat( is|'s)? (your|the) (knowledge base|corpus|data ?set|source of truth)\b"),
    ("citations",    r"\b(can|do|will) you (cite|provide|give|show)( me)? (your |the )?(sources?|citations?|references?)\b"),
    ("citations",    r"\bare your answers (cited|sourced|grounded)\b"),
    ("limits",       r"\bwhat (can('|no)?t|cannot) you (do|answer|help with)\b"),
    ("limits",       r"\bwhat are (your|the) limitations\b"),
)

_COMPILED = tuple((intent, re.compile(p, re.I)) for intent, p in _META_PATTERNS)

# The veto list. Any of these means the question is about ACME, not about me,
# even if it is phrased as a capability question. "What do you know about our
# password policy?" is the canonical case: it must go through retrieval.
_CORPUS_TERMS = (
    r"pol-\d+", r"sop-\d+", r"inc-\d+",
    r"\bpolic(y|ies)\b", r"\bsop\b", r"\bsops\b", r"\bprocedures?\b",
    r"\bincidents?\b", r"\boutages?\b", r"\bpost-?mortems?\b", r"\brunbooks?\b",
    r"\bpasswords?\b", r"\bcredentials?\b", r"\bmfa\b", r"\bauth\b",
    r"\brotation\b", r"\bencryption\b", r"\bvulnerabilit(y|ies)\b",
    r"\bteams?\b", r"\bowns?\b", r"\bowner\b", r"\bescalat", r"\bon-?call\b",
    r"\bmanages?\b", r"\bleads?\b", r"\bemployees?\b", r"\bstaff\b",
    r"\bbackups?\b", r"\bretention\b", r"\bdeploy", r"\bfailover\b",
    r"\bseverity\b", r"\bsev-?\d\b", r"\bmonitoring\b", r"\balerts?\b",
    r"\baudit\b", r"\bsla\b", r"\bcompliance\b", r"\bacme\b",
    r"\bpayment-?service\b", r"\bauth-?db\b", r"\bapi-?gateway\b",
    r"\breportingportal\b", r"\bdata ?warehouse\b", r"\bproducts?\b",
    r"\btransactions?\b", r"\bcustomers?\b", r"\bsystems?\b", r"\bdatabase\b",
)
_CORPUS_RE = re.compile("|".join(_CORPUS_TERMS), re.I)

# Long questions are compound ("what can you do, and what does SOP-02 say")
# and compound questions are not what the card answers well.
MAX_META_WORDS = 16


@dataclass(frozen=True)
class MetaMatch:
    """Why a question was routed to the card. Empty `intent` means no match."""

    intent: str
    pattern: str


def classify(question: str) -> MetaMatch | None:
    """Return a MetaMatch if `question` is about the assistant, else None.

    Conservative by construction -- see the module docstring. Every rejection
    reason is cheap and explicit so the behaviour is testable without a model.
    """
    if not question or not question.strip():
        return None
    q = question.strip()
    if len(q.split()) > MAX_META_WORDS:
        return None
    if _CORPUS_RE.search(q):
        # Talks about ACME's world. Not a question about me.
        return None
    for intent, rx in _COMPILED:
        if rx.search(q):
            return MetaMatch(intent=intent, pattern=rx.pattern)
    return None


def is_meta_question(question: str) -> bool:
    return classify(question) is not None


def answer(question: str) -> str:
    """The card. One text for every intent, on purpose.

    Splitting it per intent would mean four near-duplicate paragraphs drifting
    apart; the card is short enough that answering "which documents?" with the
    whole thing is a feature rather than noise.
    """
    return SYSTEM_CARD
