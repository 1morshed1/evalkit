## Project A: LLM Application Evaluation & Reliability Framework

### Why This Project

This is the highest-leverage project you can build. Here's why:

Every company building with LLMs has the same problem: they can't tell if their system is getting better or worse. They ship a RAG chatbot, users complain about bad answers, engineers tweak the chunking strategy or swap the model, and nobody knows if it actually improved because there's no systematic measurement.

You have built multiple RAG and agent systems. This project proves you can **evaluate** them — that you don't just build AI systems, you build AI systems that demonstrably work. That distinction is worth more than any additional feature you could add to a chatbot.

It also fills the "rigor" gap in your CV immediately. Your README will contain tables of metrics, comparison charts, and regression detection — the kind of evidence that makes a hiring manager think "this person is serious."

### What You Build

An open-source Python framework (pip-installable) that evaluates LLM applications across three dimensions: retrieval quality (for RAG), generation quality (for any LLM output), and agent reliability (for tool-using agents). It runs evaluations locally, produces structured reports, and can be plugged into CI/CD to catch regressions before deployment.

### Architecture

```
  Your Existing RAG/Agent App
       │
       │  evalkit.evaluate(questions, expected_answers)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  EvalKit Framework (Python library)                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Test Suite Runner                                       │   │
│  │                                                         │   │
│  │  Loads test cases from YAML/JSON                        │   │
│  │  Runs each through the target system                    │   │
│  │  Collects predictions + retrieved contexts              │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│           ┌─────────────┴──────────────┐                       │
│           ▼                            ▼                       │
│  ┌─────────────────┐         ┌─────────────────────┐           │
│  │  Retrieval       │         │  Generation          │          │
│  │  Evaluator       │         │  Evaluator           │          │
│  │                  │         │                      │          │
│  │  - Recall@K      │         │  - Faithfulness      │          │
│  │  - Precision@K   │         │    (grounded in      │          │
│  │  - MRR           │         │     retrieved docs?) │          │
│  │  - nDCG          │         │  - Relevance         │          │
│  │  - Hit Rate      │         │    (answers the      │          │
│  │  - Context       │         │     question?)       │          │
│  │    Relevance     │         │  - Completeness      │          │
│  └─────────────────┘         │  - Hallucination     │          │
│                              │    Detection         │          │
│           ┌──────────────────┤  - Toxicity          │          │
│           ▼                  └─────────────────────┘           │
│  ┌─────────────────┐                                           │
│  │  Agent            │                                          │
│  │  Evaluator        │                                          │
│  │                   │                                          │
│  │  - Task Completion│                                          │
│  │    Rate           │                                          │
│  │  - Tool Call      │                                          │
│  │    Accuracy       │                                          │
│  │  - Step           │                                          │
│  │    Efficiency     │                                          │
│  │  - Recovery from  │                                          │
│  │    Errors         │                                          │
│  │  - Cost per Task  │                                          │
│  └─────────────────┘                                           │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Report Generator                                       │   │
│  │                                                         │   │
│  │  - CLI table output                                     │   │
│  │  - JSON/CSV export                                      │   │
│  │  - HTML dashboard (single-file, self-contained)         │   │
│  │  - Comparison mode (config A vs config B)               │   │
│  │  - CI/CD integration (exit code 1 if regression)        │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Component | Technology | Why |
|---|---|---|
| Core Framework | Python (pip-installable package) | Your ecosystem. Installable by anyone with `pip install evalkit`. |
| LLM-as-Judge | OpenAI / Anthropic / Ollama (configurable) | Faithfulness and relevance scoring uses an LLM to judge another LLM's output. Must support any provider. |
| Test Case Format | YAML | Human-readable, version-controllable, easy to write. |
| Reporting | Rich (CLI tables) + Jinja2 (HTML report) | Beautiful terminal output for local use, HTML for sharing with stakeholders. |
| CI Integration | GitHub Actions compatible | Exit with code 1 if any metric drops below threshold. |

### Test Case Format

```yaml
# eval_suite.yaml
suite_name: "Customer Support RAG - v2.1"
target:
  type: rag                    # rag, agent, or llm
  endpoint: "http://localhost:8000/api/query"

test_cases:
  - id: "refund-policy-basic"
    question: "What is your refund policy?"
    expected_answer: "Full refund within 30 days of purchase with receipt."
    relevant_doc_ids: ["doc-policy-returns-v3"]    # ground truth for retrieval
    tags: ["policy", "refund", "high-priority"]

  - id: "refund-policy-edge"
    question: "Can I get a refund after 45 days if the product is defective?"
    expected_answer: "Defective products can be returned within 90 days under the warranty policy."
    relevant_doc_ids: ["doc-policy-warranty-v2"]
    tags: ["policy", "warranty", "edge-case"]

  - id: "pricing-question"
    question: "How much does the enterprise plan cost?"
    expected_answer: "The enterprise plan starts at $499/month with custom pricing for large teams."
    relevant_doc_ids: ["doc-pricing-2024"]
    tags: ["pricing", "sales"]

thresholds:
  retrieval_recall_at_5: 0.80
  faithfulness: 0.85
  relevance: 0.90
  hallucination_rate: 0.05    # max 5% hallucination allowed
```

### Implementation Details That Matter

**Faithfulness Evaluation (the hardest metric):**

Faithfulness measures whether the generated answer is grounded in the retrieved documents — or whether the LLM hallucinated information. This is the most important metric for any RAG system and the hardest to measure.

Your approach: decompose the answer into individual claims, then check each claim against the retrieved context.

```python
class FaithfulnessEvaluator:
    def evaluate(self, answer: str, retrieved_contexts: list[str]) -> FaithfulnessResult:
        # Step 1: Extract atomic claims from the answer
        claims = self.extract_claims(answer)
        # e.g., "The refund policy allows returns within 30 days"
        #        "A receipt is required for all returns"
        
        # Step 2: For each claim, check if it's supported by any retrieved context
        supported = 0
        unsupported_claims = []
        
        for claim in claims:
            is_supported = self.check_claim_support(claim, retrieved_contexts)
            if is_supported:
                supported += 1
            else:
                unsupported_claims.append(claim)
        
        faithfulness_score = supported / len(claims) if claims else 1.0
        
        return FaithfulnessResult(
            score=faithfulness_score,
            total_claims=len(claims),
            supported_claims=supported,
            unsupported_claims=unsupported_claims,  # these are potential hallucinations
        )
    
    def extract_claims(self, answer: str) -> list[str]:
        """Use an LLM to decompose the answer into atomic factual claims."""
        prompt = f"""Decompose the following answer into individual factual claims.
Each claim should be a single, verifiable statement.
Return a JSON array of strings.

Answer: {answer}"""
        
        response = self.judge_llm.complete(prompt)
        return json.loads(response)
    
    def check_claim_support(self, claim: str, contexts: list[str]) -> bool:
        """Use an LLM to check if the claim is supported by any context."""
        context_text = "\n---\n".join(contexts)
        prompt = f"""Given the following context passages, determine if this claim is supported.

Claim: {claim}

Context:
{context_text}

Is this claim fully supported by the context? Answer ONLY "yes" or "no"."""
        
        response = self.judge_llm.complete(prompt)
        return response.strip().lower() == "yes"
```

**Agent Evaluation (task completion + tool accuracy):**

For agent systems (which you build a lot of), evaluate not just whether the agent got the right answer, but HOW it got there:

```python
class AgentEvaluator:
    def evaluate(self, task: str, expected_result: dict, 
                 agent_trace: AgentTrace) -> AgentEvalResult:
        
        # 1. Task Completion: did the agent achieve the goal?
        task_completed = self.check_completion(task, expected_result, agent_trace.final_output)
        
        # 2. Tool Call Accuracy: did it call the right tools with correct params?
        expected_tools = expected_result.get("expected_tool_calls", [])
        actual_tools = agent_trace.tool_calls
        tool_precision = self.tool_call_precision(expected_tools, actual_tools)
        tool_recall = self.tool_call_recall(expected_tools, actual_tools)
        
        # 3. Step Efficiency: how many LLM calls did it take?
        step_count = len(agent_trace.steps)
        optimal_steps = expected_result.get("optimal_step_count", step_count)
        efficiency = min(optimal_steps / step_count, 1.0) if step_count > 0 else 0
        
        # 4. Error Recovery: did it recover from tool failures?
        tool_errors = [s for s in agent_trace.steps if s.type == "tool_error"]
        recoveries = [s for s in agent_trace.steps if s.type == "retry_after_error"]
        recovery_rate = len(recoveries) / len(tool_errors) if tool_errors else 1.0
        
        # 5. Cost: total tokens used across all LLM calls
        total_cost = sum(step.cost_usd for step in agent_trace.steps)
        
        return AgentEvalResult(
            task_completed=task_completed,
            tool_precision=tool_precision,
            tool_recall=tool_recall,
            step_efficiency=efficiency,
            error_recovery_rate=recovery_rate,
            total_steps=step_count,
            total_cost_usd=total_cost,
        )
```

**CI/CD Integration:**

```yaml
# .github/workflows/eval.yml
name: RAG Evaluation
on:
  pull_request:
    paths:
      - 'src/rag/**'           # only run when RAG code changes
      - 'config/chunking.yaml'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Start RAG service
        run: docker compose up -d
      - name: Run evaluation suite
        run: |
          pip install evalkit
          evalkit run eval_suite.yaml \
            --output results.html \
            --format ci \
            --fail-on-regression
        # Exit code 1 if any metric drops below thresholds
      - name: Upload report
        uses: actions/upload-artifact@v4
        with:
          name: eval-report
          path: results.html
```

**The Comparison Mode (A/B for configs):**

This is the killer feature. When you change your chunking strategy or swap models, run:

```bash
evalkit compare \
  --baseline results-v1.json \
  --candidate results-v2.json \
  --output comparison.html
```

Output:

```
┌─────────────────────────┬──────────┬───────────┬──────────┐
│ Metric                  │ Baseline │ Candidate │ Change   │
├─────────────────────────┼──────────┼───────────┼──────────┤
│ Retrieval Recall@5      │ 0.72     │ 0.85      │ +18.1% ✅│
│ Faithfulness            │ 0.81     │ 0.91      │ +12.3% ✅│
│ Relevance               │ 0.88     │ 0.92      │ +4.5%  ✅│
│ Hallucination Rate      │ 0.12     │ 0.04      │ -66.7% ✅│
│ Avg Latency (ms)        │ 1,240    │ 1,890     │ +52.4% ⚠️│
│ Avg Cost ($/query)      │ 0.008    │ 0.023     │ +187%  ❌│
└─────────────────────────┴──────────┴───────────┴──────────┘

Summary: Retrieval and generation quality improved significantly.
         Latency and cost increased — consider using a cheaper reranker model.
```

THIS TABLE is what goes in your README and on your CV. It instantly communicates that you measure things.

### What This Proves to Hiring Managers

- You don't just build AI systems, you know how to tell if they're working.
- You understand evaluation methodology (claim decomposition, LLM-as-judge, retrieval metrics).
- You build tools that other engineers use, not just applications.
- You can integrate AI evaluation into CI/CD pipelines, meaning you think about production reliability.

### README Structure

1. Table of evaluation metrics with definitions.
2. A "Quick Start" showing a 3-line usage example.
3. A "Case Study" section where you evaluate YOUR OWN RAG system from Shothik AI (anonymized) and show the improvement you achieved.
4. Comparison tables showing before/after metrics.

---
---