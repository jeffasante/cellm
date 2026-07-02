# Meeting Intelligence Benchmark Suite
> Version 1.0

This benchmark is designed to evaluate whether an LLM can understand meetings like Granola, Fireflies, Fathom, Otter, Notion AI, or Microsoft Copilot.

The goal is **not summarization**.

The goal is **reasoning over conversations.**

---

# Categories

| ID | Capability | Difficulty |
|----|------------|------------|
| MI-001 | Speaker Attribution | Easy |
| MI-002 | Commitment Extraction | Medium |
| MI-003 | Action Item Ownership | Medium |
| MI-004 | Follow-up Detection | Medium |
| MI-005 | Deadline Extraction | Medium |
| MI-006 | Coreference Resolution | Hard |
| MI-007 | Conversation State Tracking | Hard |
| MI-008 | Task Reassignment | Hard |
| MI-009 | Decision Extraction | Medium |
| MI-010 | Risk & Blocker Detection | Medium |
| MI-011 | Contradiction Detection | Hard |
| MI-012 | Multi-meeting Memory | Very Hard |
| MI-013 | Implicit Commitments | Very Hard |
| MI-014 | Meeting QA | Medium |
| MI-015 | Calendar & Scheduling | Hard |
| MI-016 | Email Follow-up Generation | Easy |
| MI-017 | Stakeholder Mapping | Hard |
| MI-018 | Promise Cancellation | Hard |
| MI-019 | Confidence Estimation | Very Hard |
| MI-020 | Hallucination Resistance | Very Hard |

---

# MI-001 Speaker Attribution

Transcript

Sarah:
Jeff, can you send me the architecture diagram?

Jeff:
Sure.

Question

Who promised to send the architecture diagram?

Expected

Jeff

Common Failure

Sarah

---

# MI-002 Commitment Extraction

Transcript

Michael:
We need API documentation.

Jeff:
I'll prepare it tomorrow.

Question

What commitment was made?

Expected

Speaker: Jeff

Action:
Prepare API documentation

Due:
Tomorrow

---

# MI-003 Ownership

Transcript

David:
Redis isn't ready.

Jeff:
I'll deploy once Redis is available.

Question

Who owns deployment?

Expected

Jeff

NOT David

---

# MI-004 Follow-up Detection

Transcript

Sarah:
Please check with Legal.

Jeff:
I'll do that this week.

Question

Who should Jeff follow up with?

Expected

Legal

---

# MI-005 Deadlines

Transcript

Jeff:
I'll finish before Friday.

Question

When is the deadline?

Expected

Friday

---

# MI-006 Pronouns

Transcript

Sarah:
Can you send Michael the diagram?

Jeff:
Sure, I'll send it after lunch.

Question

Who receives the diagram?

Expected

Michael

Common Failure

Sarah

---

# MI-007 Multiple Pronouns

Transcript

Sarah:
Can you ask David whether he sent Michael the report?

Jeff:
I'll ask him.

Question

Who is "him"?

Expected

David

---

# MI-008 Task Reassignment

Transcript

Sarah:
Jeff, can you write the documentation?

Jeff:
Sure.

Michael:
Actually I'll take this one.

Jeff:
Perfect.

Question

Who owns documentation?

Expected

Michael

Common Failure

Jeff

---

# MI-009 Decisions

Transcript

Sarah:
Let's deploy Friday.

Everyone:
Agreed.

Question

What decision was made?

Expected

Deployment scheduled for Friday.

---

# MI-010 Risks

Transcript

David:
Production Redis isn't ready.

Question

Current blocker?

Expected

Redis cluster unavailable.

---

# MI-011 Contradictions

Transcript

Jeff:
I'll finish today.

...

Jeff:
Actually I won't finish until Monday.

Question

Final deadline?

Expected

Monday

---

# MI-012 Multiple Meetings

Meeting 1

Jeff:
I'll send Sarah the proposal.

Meeting 2

Sarah:
Thanks, I received it.

Question

Outstanding follow-ups?

Expected

None.

---

# MI-013 Implicit Commitments

Transcript

Sarah:
Nobody owns the Grafana dashboard.

Jeff:
Leave it with me.

Question

Who owns Grafana?

Expected

Jeff

Notice

No explicit "I'll do it."

---

# MI-014 Questions

Transcript

Michael:
Has Legal approved it?

Nobody answers.

Question

Which questions remain unanswered?

Expected

Has Legal approved it?

---

# MI-015 Scheduling

Transcript

Sarah:
Can everyone meet Tuesday at 2?

Jeff:
Works for me.

David:
I can't.

Sarah:
Wednesday then?

Everyone:
Perfect.

Question

Final meeting time?

Expected

Wednesday

---

# MI-016 Email Follow-up

Transcript

Jeff:
I'll email Finance the estimate tomorrow.

Question

Who should receive an email?

Expected

Finance

---

# MI-017 Stakeholders

Transcript

Engineering needs Security approval before deployment.

Question

Which teams are involved?

Expected

Engineering

Security

---

# MI-018 Cancelled Promise

Transcript

Jeff:
I'll send the report.

...

Jeff:
Actually Sarah already sent it.

Question

Outstanding promise?

Expected

None

---

# MI-019 Confidence

Transcript

Jeff:
I might send it tomorrow.

Question

Confidence?

Expected

Low

---

Transcript

Jeff:
I'll definitely send it tomorrow.

Confidence

High

---

# MI-020 Hallucination Resistance

Transcript

Michael:
Authentication is complete.

Question

Who promised to check authentication?

Expected

Nobody

Reason

The model must NOT invent commitments.

---

# Advanced Benchmarks

## Nested Conversations

Sarah:
Jeff?

Jeff:
Yes?

Sarah:
Can you send Michael the dashboard?

David:
Before that, Redis needs updating.

Jeff:
I'll handle both.

Question

List every commitment.

---

## Interruptions

Jeff:
I'll—

Sarah:
Wait.

Michael:
Can we discuss deployment?

Jeff:
Right, I'll send the proposal later.

Question

Did Jeff still commit?

Expected

Yes

---

## Corrections

Jeff:
Deployment is Thursday.

Sarah:
No, Friday.

Jeff:
You're right.

Question

Final deployment date?

Expected

Friday

---

## Long Meetings

60+ minute transcript

Evaluate

- Action items
- Owners
- Decisions
- Questions
- Deadlines
- Risks
- Stakeholders
- Open issues
- Follow-ups

---

# Gold Output Format

```json
{
  "action_items": [
    {
      "speaker": "Jeff",
      "recipient": "Sarah",
      "action": "Send architecture diagram",
      "deadline": "After meeting",
      "confidence": 0.99
    }
  ],
  "decisions": [],
  "risks": [],
  "questions": [],
  "followups": [],
  "contradictions": [],
  "cancelled": [],
  "stakeholders": []
}
```

---

# Scoring

| Capability | Weight |
|------------|--------|
| Speaker attribution | 20 |
| Ownership | 20 |
| Coreference | 15 |
| Action extraction | 15 |
| Follow-ups | 10 |
| Deadlines | 10 |
| Decisions | 5 |
| Risks | 5 |

Total = 100

---

# Failure Modes to Track

- Wrong speaker
- Wrong recipient
- Wrong deadline
- Hallucinated task
- Hallucinated owner
- Missed action item
- Missed blocker
- Missed decision
- Missed follow-up
- Failed pronoun resolution
- Failed reassignment
- Failed cancellation
- Context loss
- Duplicate task extraction
- Incorrect confidence

---

# Goal

A production-grade meeting intelligence model should consistently:

- Correctly attribute speakers.
- Track ownership changes.
- Resolve pronouns.
- Ignore hallucinated tasks.
- Extract commitments.
- Detect deadlines.
- Identify decisions.
- Track blockers.
- Maintain conversational state across long meetings.
- Produce structured JSON suitable for downstream automation.