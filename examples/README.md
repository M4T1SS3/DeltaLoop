# DeltaLoop Example

## Customer Support Agent

**File**: `customer_support_agent.py`

A realistic example showing how to train a customer support AI agent using DeltaLoop.

### Scenario

E-commerce customer support agent that handles:
- Order tracking and status
- Refund and return requests
- Product questions and recommendations
- Technical support and troubleshooting
- Complaints and escalations
- Warranty claims
- Policy questions

### Problem

Base models struggle with:
- Knowing when to use which tool
- Following company policies (30-day returns, warranty rules)
- Properly escalating to humans
- Consistent support quality

### Solution

Use DeltaLoop to fine-tune on successful support interactions, making the agent better without extensive prompt engineering.

---

## Usage

### Generate Support Data Only (No Training)

```bash
python3 examples/customer_support_agent.py --no-train
```

Creates realistic customer support interactions without requiring ML dependencies.

### Run Complete Pipeline (Training Included)

```bash
# Basic run (SFT - Supervised Fine-Tuning)
python3 examples/customer_support_agent.py

# DPO training (learns from both success AND failures!)
python3 examples/customer_support_agent.py --training-method dpo

# More training for better results
python3 examples/customer_support_agent.py --steps 200 --training-method dpo

# More interactions
python3 examples/customer_support_agent.py --num-interactions 150

# Custom model
python3 examples/customer_support_agent.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## What It Creates

### Support Interactions
**Location**: `data/customer_support/interactions.jsonl`

30 realistic scenarios including:
- "Where is my order ORD-12346?"
- "I want a refund for order ORD-12345"
- "Is the USB-C Hub compatible with MacBook Pro?"
- "My keyboard isn't connecting via Bluetooth"
- "I want to speak to a manager NOW!"

### Training Pipeline

**SFT Mode (default):**
1. **Filters** successful interactions only
2. **Deduplicates** similar queries
3. **Trains** LoRA adapter
4. **Evaluates** improvement

**DPO Mode (--training-method dpo):**
1. **Matches** successful with failed interactions
2. **Creates** preference pairs (chosen vs rejected)
3. **Trains** with Direct Preference Optimization
4. **Evaluates** improvement

DPO learns from BOTH success and failure - teaching the model what's better, not just what's good!

### Trained Adapter
**Location**: `data/customer_support/pipeline_runs/support_agent_*/adapter`

Fine-tuned model that's better at:
- Using the right tools (check_order_status, process_refund, etc.)
- Following company policies
- Escalating appropriately
- Providing helpful, consistent responses

---

## Requirements

### For Data Generation Only
```bash
# No dependencies needed!
python3 examples/customer_support_agent.py --no-train
```

### For Full Pipeline (Training + Evaluation)
```bash
pip install transformers torch peft trl datasets
```

---

## Expected Results

After training, the agent shows:
- ✓ Better tool usage (+40-50%)
- ✓ Better policy adherence
- ✓ Better escalation handling
- ✓ More consistent responses

Example improvement:
```
Task: tool_use_accuracy
  Baseline: 60%
  Adapted:  85%  (+41.7%)

Overall Improvement: +30-40%
```

---

## Next Steps

### 1. Use the Trained Agent

```python
from deltaloop import load_model

model, tokenizer = load_model(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter="data/customer_support/pipeline_runs/support_agent_*/adapter"
)

# Now use in your support system
```

### 2. Review the Data

```bash
# View interactions
cat data/customer_support/interactions.jsonl | python3 -m json.tool | less

# Check training data
cat data/customer_support/pipeline_runs/*/train.jsonl | python3 -m json.tool | less
```

### 3. Check Results

```bash
cat data/customer_support/pipeline_runs/*/results.json | python3 -m json.tool
```

---

## Customization

### Add Your Own Scenarios

Edit the `scenarios` list in `generate_support_interactions()` to add company-specific situations:

```python
scenarios.append({
    "prompt": "Your custom customer query",
    "output": "Agent's ideal response with tool calls",
    "success": True,
    "tool_calls": [{"tool": "your_tool", "input": "params"}]
})
```

### Use Real Logs

Replace synthetic generation with your actual support logs:

```python
from deltaloop import run_pipeline

result = run_pipeline(
    logs_path="data/your_real_support_logs.jsonl",  # Your actual logs
    training_steps=200
)
```

---

## Why This Example?

This example is realistic because:
- **Real tool usage** - check_order_status, process_refund, create_ticket
- **Real policies** - 30-day returns, warranty coverage
- **Real edge cases** - invalid orders, out of stock, angry customers
- **Real escalations** - knows when to involve humans

Much better than toy "what is 2+2" examples!
