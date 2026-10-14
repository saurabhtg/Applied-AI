"""
vol3_agents/part14_working_memory/working_memory.py

The token accountant.

A 1M-token window doesn't mean you should fill it. Accuracy drops long
before the window does, and every turn re-reads everything you kept. This
module gives you:
    - A running token estimate that calibrates itself against the
      usage.input_tokens the API sends back with every response.
    - A bounded history that compacts in bulk, rarely. When the buffer
      crosses its high-water mark, the oldest unpinned entries are dropped
      (or summarized via a callback) until it's down to a low-water mark.
      Then it stays append-only until the next sweep, which keeps the
      prompt cache warm in between.
    - Pinned entries that no sweep ever touches.

The principle: working memory is finite. Treat it like RAM, not disk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """A rough count, about 3.5 characters per token.

    Deliberately not tiktoken: that's OpenAI's tokenizer and it undercounts
    Claude. For an exact number use client.messages.count_tokens(), or let
    WorkingMemory.observe_usage() correct this estimate from real responses.
    """
    return max(1, round(len(text) / 3.5))


@dataclass
class MemoryEntry:
    role: str  # "user" | "assistant" | "system" | "summary"
    content: str
    tokens: int


@dataclass
class WorkingMemory:
    """
    A bounded, token-aware conversation buffer for agents.

    When the buffer crosses its budget, the oldest unpinned entries are
    dropped (or summarized via a callback) in one sweep, down to
    `compact_to` of the budget. System entries and pinned entries are
    never touched.
    """

    max_tokens: int = 64_000
    reserve_tokens: int = 16_000  # Headroom for the next response, thinking included.
    compact_to: float = 0.5       # Low-water mark, as a fraction of the usable budget.
    entries: list[MemoryEntry] = field(default_factory=list)
    pinned_indices: set[int] = field(default_factory=set)
    summarize_fn: Callable[[list[MemoryEntry]], str] | None = None
    count_fn: Callable[[str], int] = estimate_tokens
    compactions: int = 0
    _scale: float = 1.0  # actual / estimated, learned from observe_usage()

    @property
    def usable_tokens(self) -> int:
        return self.max_tokens - self.reserve_tokens

    @property
    def total_tokens(self) -> int:
        return round(sum(e.tokens for e in self.entries) * self._scale)

    @property
    def available_tokens(self) -> int:
        return self.usable_tokens - self.total_tokens

    def add(self, role: str, content: str, pinned: bool = False) -> None:
        """Append an entry. Sweep if the budget is exceeded."""
        self.entries.append(MemoryEntry(role=role, content=content, tokens=self.count_fn(content)))
        if pinned:
            self.pinned_indices.add(len(self.entries) - 1)
        if self.total_tokens > self.usable_tokens:
            self._compact()

    def pin(self, index: int) -> None:
        """Pin an entry so it survives compaction."""
        if 0 <= index < len(self.entries):
            self.pinned_indices.add(index)

    def observe_usage(self, input_tokens: int) -> None:
        """Calibrate the estimate against what the API actually counted.

        Call it right after the API call, before you add the reply, with
        response.usage.input_tokens. Tool definitions count too, so the
        scale comes out slightly high, which errs on the safe side.
        """
        estimated = sum(e.tokens for e in self.entries)
        if estimated > 0:
            self._scale = max(1.0, input_tokens / estimated)

    def system(self) -> str:
        """The system prompt: every system entry, in order."""
        return "\n\n".join(e.content for e in self.entries if e.role == "system")

    def render(self) -> list[dict]:
        """The conversation in API-message format, system entries excluded.

        A summary becomes a user message, so it reaches the model and the
        conversation still starts on a user turn.
        """
        messages = []
        for e in self.entries:
            if e.role == "system":
                continue
            if e.role == "summary":
                messages.append({"role": "user", "content": f"[Summary of the conversation so far] {e.content}"})
            else:
                messages.append({"role": e.role, "content": e.content})
        return messages

    def _compact(self) -> None:
        """Sweep the oldest unpinned entries until under the low-water mark."""
        target = self.usable_tokens * self.compact_to
        evictable = [
            i for i, e in enumerate(self.entries)
            if i not in self.pinned_indices and e.role != "system"
        ]
        if not evictable:
            logger.warning(
                "WorkingMemory over budget (%d/%d) but everything left is pinned",
                self.total_tokens, self.usable_tokens,
            )
            return

        victims: list[int] = []
        freed = 0
        for i in evictable[:-1]:  # never evict the entry just added
            victims.append(i)
            freed += round(self.entries[i].tokens * self._scale)
            if self.total_tokens - freed <= target:
                break
        # Stop on a turn boundary, so the kept conversation starts with the user.
        for i in evictable[len(victims):-1]:
            if self.entries[i].role == "user":
                break
            victims.append(i)

        if not victims:
            return

        evicted = [self.entries[i] for i in victims]
        replacement: list[MemoryEntry] = []
        if self.summarize_fn is not None:
            summary = self.summarize_fn(evicted)
            replacement = [MemoryEntry(role="summary", content=summary, tokens=self.count_fn(summary))]

        first = victims[0]
        victim_set = set(victims)
        kept: list[MemoryEntry] = []
        new_pins: set[int] = set()
        for i, e in enumerate(self.entries):
            if i == first:
                kept.extend(replacement)
            if i in victim_set:
                continue
            if i in self.pinned_indices:
                new_pins.add(len(kept))
            kept.append(e)

        self.entries = kept
        self.pinned_indices = new_pins
        self.compactions += 1
        logger.info(
            "WorkingMemory compacted: %d entries out, %s, now %d/%d tokens",
            len(evicted), "summarized" if replacement else "dropped",
            self.total_tokens, self.usable_tokens,
        )
