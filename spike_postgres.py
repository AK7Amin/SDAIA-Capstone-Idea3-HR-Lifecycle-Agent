"""Spike: prove PostgresSaver semantics BEFORE building on them.

Questions this must answer (lessons from the previous capstone, where wrong
assumptions about SqliteSaver cost a rebuild):
1. How is PostgresSaver constructed? (context manager? from_conn_string? setup()?)
2. Does interrupt() + Command(resume=...) survive a NEW connection (simulating
   a new process days later)?
3. Does state written before the interrupt survive intact?

Run: python spike_postgres.py   (expects postgres on localhost:5433, pw=capstone)
Delete after the PRD locks in the findings.
"""
import sys

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

DSN = "postgresql://postgres:capstone@localhost:5433/hr_agent"


class S(TypedDict, total=False):
    candidate: str
    plan: str
    approved: str
    status: str


def build(checkpointer):
    g = StateGraph(S)
    g.add_node("draft", lambda s: {"plan": f"plan-for-{s['candidate']}", "status": "drafted"})

    def gate(s):
        decision = interrupt({"ask": "approve?"})
        return {"approved": decision, "status": "resumed"}

    g.add_node("gate", gate)
    g.add_node("finish", lambda s: {"status": f"done-{s['approved']}"})
    g.add_edge(START, "draft")
    g.add_edge("draft", "gate")
    g.add_edge("gate", "finish")
    g.add_edge("finish", END)
    return g.compile(checkpointer=checkpointer)


def main():
    # Q1: construction. Try from_conn_string as a context manager (docs pattern).
    with PostgresSaver.from_conn_string(DSN) as saver:
        saver.setup()          # creates tables; idempotent?
        saver.setup()          # run twice to confirm idempotency
        graph = build(saver)
        cfg = {"configurable": {"thread_id": "spike-1"}}
        out = graph.invoke({"candidate": "sara"}, cfg)
        print("PHASE1 status:", out.get("status"), "| interrupted:", "__interrupt__" in out)

    # Q2: resume on a FRESH connection (new process simulation).
    with PostgresSaver.from_conn_string(DSN) as saver2:
        graph2 = build(saver2)
        cfg = {"configurable": {"thread_id": "spike-1"}}
        out2 = graph2.invoke(Command(resume="yes"), cfg)
        print("PHASE2 status:", out2.get("status"))
        print("PHASE2 plan survived:", out2.get("plan"))
        assert out2.get("status") == "done-yes", "resume across connections FAILED"
        assert out2.get("plan") == "plan-for-sara", "pre-interrupt state LOST"
    print("SPIKE PASSED: from_conn_string CM + setup() + cross-connection resume all work")


if __name__ == "__main__":
    sys.exit(main())
