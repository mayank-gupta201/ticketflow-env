# TicketFlowEnv

TicketFlowEnv is a realistic OpenEnv-compatible benchmark where an AI agent must classify, investigate, resolve, and close customer support tickets while following deterministic business policy.

## Why This Environment Matters

Customer support is a high-value, operationally meaningful agent domain. Real support agents do not just answer questions: they must follow policy, choose safe actions, avoid harmful resolutions, ask for more information when ambiguity remains, and close tickets only when the workflow is complete. TicketFlowEnv captures those constraints in a lightweight, reproducible benchmark designed for evaluation, training, and hackathon deployment.

## Real-World Utility

This environment models a common enterprise workflow:

- A customer opens a support ticket.
- The agent must classify the issue type.
- The agent must choose an appropriate operational action.
- The agent must stay within policy boundaries.
- The agent must send a customer-facing reply.
- The agent must only close the ticket when closure is operationally correct.

That combination makes the environment useful for testing real agent behavior instead of toy-game performance. It rewards policy compliance, penalizes harmful resolution decisions, and uses hidden internal state to support deterministic grading.

## Project Structure

```text
ticketflow_env/
|-- ticketflow_env/
|   |-- __init__.py
|   |-- models.py
|   |-- client.py
|   `-- server/
|       |-- __init__.py
|       |-- app.py
|       `-- environment.py
|-- server/                  # OpenEnv-compliant entry point shim
|   |-- __init__.py
|   `-- app.py
|-- tasks.py
|-- grader.py
|-- reward.py
|-- policy.py
|-- inference.py
|-- openenv.yaml
|-- requirements.txt
|-- Dockerfile
|-- README.md
|-- pyproject.toml
`-- tests/
    `-- test_env.py
```

## Action Space

TicketFlowEnv uses a structured action space with the following allowed `action_type` values:

```python
AVAILABLE_ACTIONS = [
    "classify_issue",
    "request_more_info",
    "approve_refund",
    "deny_refund",
    "offer_replacement",
    "offer_store_credit",
    "escalate_to_human",
    "send_customer_reply",
    "close_ticket",
]
```

Each action is represented by a typed `TicketFlowAction` model:

```python
{
  "action_type": "classify_issue",
  "response_text": null,
  "metadata": {"label": "damaged_item"}
}
```

## Observation Space

The agent only sees the customer-facing context needed to act:

- `ticket_id`
- `customer_message`
- `customer_tier`
- `order_status`
- `order_value`
- `order_age_days`
- `conversation_history`
- `available_actions`
- `current_status`
- `last_action_result`

The observation intentionally hides grading truth such as the correct resolution, hidden policy flags, and internal workflow success markers.

## State And Hidden Truth

The hidden `TicketFlowState` model stores the deterministic internal truth used for transitions and grading:

- `ticket_id`
- `issue_type`
- `correct_resolution`
- `refund_allowed`
- `replacement_allowed`
- `escalation_required`
- `classification_done`
- `reply_sent`
- `resolved`
- `failed`
- `max_steps`

The implementation also tracks hidden operational state such as the chosen resolution action, internal step count, closure state, and cumulative reward.

## Benchmark Tasks

### 1. `easy_refund`

Difficulty: Easy

Customer message:

> My package arrived broken and I want a refund.

Expected strong flow:

1. `classify_issue` with `label="damaged_item"`
2. `approve_refund`
3. `send_customer_reply`
4. `close_ticket`

### 2. `account_access_ambiguity`

Difficulty: Medium

Customer message:

> I can't access my account and can't see my order.

Expected strong flow:

1. `classify_issue` with `label="account_access_issue"`
2. `request_more_info`
3. `send_customer_reply`
4. Optional `escalate_to_human` only if used reasonably

This task penalizes premature refund actions or irrelevant resolutions.

### 3. `out_of_policy_refund`

Difficulty: Hard

Customer message:

> It's been 35 days and I want a refund because I changed my mind.

Expected strong flow:

1. `classify_issue` with `label="refund_request"`
2. `deny_refund` or `offer_store_credit` or `escalate_to_human`
3. `send_customer_reply`
4. `close_ticket`

This task strongly rewards policy compliance and penalizes approving a refund outside the allowed window.

## Reward Function

TicketFlowEnv uses dense, deterministic reward shaping across the full trajectory.

Positive signals:

- `+0.20` correct classification
- `+0.25` correct next workflow action
- `+0.20` policy-compliant action
- `+0.25` correct resolution action
- `+0.10` helpful customer reply

Negative signals:

- `-0.30` invalid action
- `-0.40` harmful action
- `-0.10` repeated redundant action
- `-0.15` unnecessary escalation
- `-0.25` closing an unresolved ticket

Rewards are clamped to `[-1.0, 1.0]`.

## Grader

The grader is fully deterministic and does not use an LLM.

Each task grader returns:

```python
{
  "score": 0.75,
  "details": {
    "correct_classification": True,
    "followed_policy": True,
    "used_valid_resolution": True,
    "reply_sent": True,
    "closed_ticket": False
  }
}
```

Scoring depends on:

- the full action history
- hidden state flags
- policy correctness
- task-specific success criteria

## Episode Rules

- `MAX_STEPS = 6`
- An episode ends when the ticket is resolved, failed badly, closed, or reaches max steps
- Closing a ticket too early ends the episode with a low score
- The environment supports deterministic task selection with `reset(task_id=...)`

## Setup

### Local Python Setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Local Run

Start the environment server:

```bash
uvicorn ticketflow_env.server.app:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Docker Build And Run

Build:

```bash
docker build -t ticketflow-env .
```

Run:

```bash
docker run --rm -p 8000:8000 ticketflow-env
```

## How To Validate

Run OpenEnv structural validation:

```bash
openenv validate
openenv validate --verbose
```

Run the unit tests:

```bash
python -m pytest tests/ -v
```

You can also manually step the environment from Python:

```python
from ticketflow_env.server.environment import TicketFlowEnvironment
from ticketflow_env.models import TicketFlowAction

env = TicketFlowEnvironment()
obs = env.reset(task_id="easy_refund")
result = env.step(TicketFlowAction(action_type="classify_issue", metadata={"label": "damaged_item"}))
print(result.reward, result.done, result.observation.current_status)
```

## How To Run Inference

The baseline runner uses the OpenAI Python client for action generation and safely falls back to a deterministic heuristic policy if the model output is invalid or unavailable.

Required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- `IMAGE_NAME` optional

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_xxx"
python inference.py
```

## Example Output

```text
[START] task=easy_refund env=TicketFlowEnv model=meta-llama/Llama-3.1-8B-Instruct
[STEP] step=1 action={"action_type": "classify_issue", ...} reward=0.65 done=false error=null
[STEP] step=2 action={"action_type": "approve_refund", ...} reward=0.70 done=false error=null
[STEP] step=3 action={"action_type": "send_customer_reply", ...} reward=0.55 done=false error=null
[STEP] step=4 action={"action_type": "close_ticket", ...} reward=0.45 done=true error=null
[END] success=true steps=4 score=1.000 rewards=0.65,0.70,0.55,0.45
baseline_average_score=1.000
```

## Hugging Face Deployment Notes

TicketFlowEnv is designed to be lightweight and Space-friendly:

- no external APIs are required by the environment server
- no database is required
- no GPU is required
- memory and CPU footprint fit comfortably within a `2 vCPU / 8 GB RAM` deployment target
- the server runs as a single FastAPI app on port `8000`
- deterministic tasks and graders make remote evaluation reproducible

This design is well suited for Hugging Face Spaces and the OpenEnv deployment workflow.
