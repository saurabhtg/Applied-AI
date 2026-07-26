"""
examples/oncall_rag/corpus.py

Small fixture corpus for the on-call RAG example.

A real internal knowledge base runs to thousands of pages: runbooks,
postmortems, architecture docs, Slack threads, and the occasional wiki
page someone wrote in 2021 and never updated. We've distilled the kind
of operational knowledge a 3 a.m. on-call engineer actually reaches for
into fifteen passages.

These are written for the example, not lifted from a real company. Note
two things planted on purpose, because they're where naive RAG breaks:

  1. Exact tokens an embedding model has never learned to place:
     ERR_5521, OOMKilled, CrashLoopBackOff, ImagePullBackOff,
     MAX_POOL_SIZE, payments-api, checkout-worker. BM25 nails these;
     pure vectors return generic mush.

  2. A stale runbook (rb_010) that has been superseded by a newer
     postmortem (rb_011). They're about the same topic, so semantic
     search ranks them roughly equally — and happily hands you the
     dangerous, outdated one. Recency/metadata is the fix, not vectors.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CorpusDoc:
    id: str
    source: str
    text: str


ONCALL_CORPUS: list[CorpusDoc] = [
    CorpusDoc(
        id="rb_001",
        source="Runbook: Deployments",
        text=(
            "To roll back a bad deployment, use the deploy tool: run "
            "`deployctl rollback <service> --to-previous`. This reverts the "
            "service to the last known-good release recorded in the deploy "
            "ledger. Rollbacks are safe to run at any time and do not require "
            "an approval. Always roll back first and investigate after — a "
            "live incident is not the time to debug a forward fix."
        ),
    ),
    CorpusDoc(
        id="rb_002",
        source="Reference: Error Codes",
        text=(
            "ERR_5521 is raised by payments-api when the upstream payment "
            "gateway does not respond within the configured timeout. It is an "
            "upstream timeout, not a bug in our code. First check the gateway "
            "status page, then confirm whether the timeout is widespread or "
            "limited to one merchant. ERR_5521 is transient in most cases and "
            "clears once the gateway recovers; persistent ERR_5521 should be "
            "escalated to the Payments team."
        ),
    ),
    CorpusDoc(
        id="rb_003",
        source="Runbook: Kubernetes Pods",
        text=(
            "A pod terminated with reason OOMKilled was killed by the kernel "
            "for exceeding its memory limit. Confirm with "
            "`kubectl describe pod <name>` and look for 'OOMKilled' under Last "
            "State. This commonly hits checkout-worker during large batch jobs. "
            "Short term, restart is automatic. Real fix: raise the memory limit "
            "in the deployment manifest or find the leak. Repeated OOMKilled "
            "events on the same pod mean the limit is wrong, not the node."
        ),
    ),
    CorpusDoc(
        id="rb_004",
        source="Runbook: Kubernetes Pods",
        text=(
            "CrashLoopBackOff means a pod starts, crashes, and Kubernetes keeps "
            "restarting it with growing back-off delays. It is a symptom, not a "
            "cause. Get the real error with `kubectl logs <pod> --previous` to "
            "read the logs from the crashed instance. The usual causes are a bad "
            "config value, a missing secret or env var, or a failed database "
            "connection at startup. Do not just delete the pod — it will "
            "CrashLoopBackOff again until the underlying cause is fixed."
        ),
    ),
    CorpusDoc(
        id="rb_005",
        source="Architecture: payments-api",
        text=(
            "payments-api is the service that brokers all card and UPI payment "
            "requests to upstream gateways. It owns the payments database and is "
            "fronted by the api-gateway. Its dashboards live in Grafana under "
            "'Payments / payments-api'. The team that owns it is Payments-Core. "
            "It is a tier-1 service: any outage is automatically a Sev-1."
        ),
    ),
    CorpusDoc(
        id="rb_006",
        source="Runbook: Incident Response",
        text=(
            "When something is broken badly enough to affect customers, declare "
            "a Sev-1. Type `/incident declare sev1` in Slack, which pages the "
            "on-call Incident Commander and opens a dedicated incident channel "
            "and a video bridge. The first job is not to fix it — it is to "
            "mitigate (roll back, fail over, or flag off) and keep a running "
            "timeline. The IC coordinates; engineers investigate; nobody makes "
            "silent changes during a Sev-1."
        ),
    ),
    CorpusDoc(
        id="rb_007",
        source="Runbook: payments-api / Database",
        text=(
            "For the connection from payments-api to its primary database, set "
            "the connection timeout to 30s and retry up to 3 times with "
            "exponential back-off. The pool is sized by MAX_POOL_SIZE, which "
            "defaults to 20. If you see 'pool exhausted' errors under load, the "
            "first lever is MAX_POOL_SIZE — but raising it past the database's "
            "own connection limit just moves the failure, so check both."
        ),
    ),
    CorpusDoc(
        id="rb_008",
        source="Runbook: Latency",
        text=(
            "When the service gets slow under heavy traffic, look at p99 latency "
            "first, not the average — the average hides the tail that customers "
            "actually feel. Check whether the slowdown tracks request volume "
            "(capacity problem) or is flat (a dependency is slow). Common causes "
            "are database connection pool exhaustion, an N+1 query pattern, or a "
            "slow downstream call with no timeout. Scaling up rarely fixes a "
            "latency problem caused by a single slow dependency."
        ),
    ),
    CorpusDoc(
        id="rb_009",
        source="Runbook: On-call & Escalation",
        text=(
            "Primary on-call is paged first via PagerDuty. If the primary does "
            "not acknowledge within 5 minutes, the page escalates to the "
            "secondary, and after a further 10 minutes to the engineering "
            "manager. You are never expected to fix everything alone — pull in "
            "the owning team early. Escalating is not a failure; a prolonged "
            "outage because nobody pulled the right person in is."
        ),
    ),
    CorpusDoc(
        id="rb_010",
        source="Runbook: Pod Recovery (DEPRECATED, 2024)",
        text=(
            "To recover a stuck payments-api pod, SSH into the node and run "
            "`docker restart` on the container directly. This forces an "
            "immediate restart without waiting for Kubernetes to reschedule. "
            "Use this when a pod is wedged and you need it back fast."
        ),
    ),
    CorpusDoc(
        id="rb_011",
        source="Postmortem: INC-4821 (2026-03)",
        text=(
            "NEVER manually restart a payments-api pod by SSHing into the node "
            "or running docker restart. INC-4821 was caused by exactly this: a "
            "hard restart killed an in-flight write and corrupted the "
            "write-ahead log, turning a 2-minute blip into a 40-minute outage. "
            "The correct recovery is `deployctl drain payments-api` first, which "
            "lets in-flight requests finish, then let Kubernetes reschedule. This "
            "supersedes the old pod-recovery runbook."
        ),
    ),
    CorpusDoc(
        id="rb_012",
        source="Runbook: Kubernetes Pods",
        text=(
            "ImagePullBackOff means Kubernetes cannot pull the container image. "
            "It is almost never a code problem. Check, in order: the image tag "
            "actually exists in the registry, the registry pull secret is "
            "present and not expired, and the node has network access to the "
            "registry. A typo in the image tag during a deploy is the single "
            "most common cause of ImagePullBackOff."
        ),
    ),
    CorpusDoc(
        id="rb_013",
        source="Runbook: Disk & Storage",
        text=(
            "'No space left on device' on a node usually means logs or temp "
            "files filled the disk. Find the offender with `du -sh /var/* | sort "
            "-h`. Most often it is unrotated application logs. Rotate or "
            "truncate them, confirm logrotate is configured, and check for a "
            "process writing an unbounded file. A full disk can cascade: it "
            "makes pods on that node fail in confusing, unrelated-looking ways."
        ),
    ),
    CorpusDoc(
        id="rb_014",
        source="Runbook: Feature Flags",
        text=(
            "If a newly shipped feature is causing errors, you do not need a "
            "deploy to turn it off. Run `flagctl disable <flag-name>` to kill "
            "the feature for all users immediately. Flag changes take effect "
            "within seconds and need no rollback. Turning a flag off is the "
            "fastest mitigation available — reach for it before a code rollback "
            "when the bad change is behind a flag."
        ),
    ),
    CorpusDoc(
        id="rb_015",
        source="Runbook: TLS & Certificates",
        text=(
            "HTTP 526 errors from the edge mean the origin's TLS certificate is "
            "invalid or expired. Check the certificate's expiry with "
            "`openssl s_client -connect <host>:443`. Certificates are renewed "
            "automatically by cert-manager; a 526 usually means the renewal job "
            "failed silently. Re-trigger the renewal and confirm the new "
            "certificate's not-after date is in the future."
        ),
    ),
]
