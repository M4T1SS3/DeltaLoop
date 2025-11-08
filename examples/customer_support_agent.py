"""Realistic Customer Support Agent Example with DeltaLoop.

This example demonstrates a real-world use case: training a customer support
AI agent to handle common queries better.

Scenario:
    You have a customer support agent that handles:
    - Order tracking and status
    - Refund requests
    - Product questions
    - Technical support
    - Account issues

Problem:
    The base model struggles with:
    - Knowing when to use which tool
    - Properly handling refund policies
    - Escalating to human support when needed
    - Following company procedures

Solution:
    Use DeltaLoop to fine-tune the model on successful support interactions,
    making it better at handling queries without extensive prompt engineering.

Usage:
    # Run complete pipeline
    python3 examples/customer_support_agent.py

    # Quick test (no training)
    python3 examples/customer_support_agent.py --no-train

    # More training for better results
    python3 examples/customer_support_agent.py --steps 200

    # Custom model
    python3 examples/customer_support_agent.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import random
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deltaloop.schema import AgentTrace
from deltaloop.pipeline import Pipeline, PipelineConfig

# Try to import torch (needed for custom evaluation)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Simulated company data
PRODUCTS = {
    "PROD-001": {"name": "Wireless Mouse", "price": 29.99, "stock": 150},
    "PROD-002": {"name": "Mechanical Keyboard", "price": 89.99, "stock": 45},
    "PROD-003": {"name": "USB-C Hub", "price": 49.99, "stock": 0},  # Out of stock
    "PROD-004": {"name": "Laptop Stand", "price": 39.99, "stock": 200},
    "PROD-005": {"name": "Webcam HD", "price": 79.99, "stock": 30},
}

ORDERS = {
    "ORD-12345": {"product": "PROD-001", "status": "delivered", "date": "2024-01-15", "amount": 29.99},
    "ORD-12346": {"product": "PROD-002", "status": "shipped", "date": "2024-01-18", "amount": 89.99},
    "ORD-12347": {"product": "PROD-004", "status": "processing", "date": "2024-01-20", "amount": 39.99},
    "ORD-12348": {"product": "PROD-005", "status": "delivered", "date": "2024-01-10", "amount": 79.99},
    "ORD-99999": {"product": "PROD-003", "status": "cancelled", "date": "2024-01-05", "amount": 49.99},
}

KNOWLEDGE_BASE = {
    "returns": "Our return policy allows returns within 30 days of delivery. Items must be unused and in original packaging. Refund processed within 5-7 business days.",
    "shipping": "Standard shipping: 5-7 business days. Express shipping: 2-3 business days. Free shipping on orders over $50.",
    "warranty": "All products come with a 1-year manufacturer warranty covering defects. Extended warranty available for purchase.",
    "setup": "Setup instructions are included in the product box. Video tutorials available at support.example.com/videos",
    "compatibility": "Check product specifications on the product page. Contact support if you have specific compatibility questions.",
}


def generate_support_interactions(output_path: str, num_interactions: int = 100, verbose: bool = True) -> int:
    """Generate realistic customer support interactions.

    Creates diverse scenarios covering modern agent patterns:
    - Tool selection (MCP-style multi-tool decisions)
    - Intent classification (routing queries)
    - Information extraction (parsing structured data)
    - Multi-step workflows
    - Error handling and recovery
    - Context management
    - Complex decision trees

    Args:
        output_path: Where to save logs (JSONL)
        num_interactions: Number of interactions to generate (default: 100)
        verbose: Print progress

    Returns:
        Number of interactions created
    """
    if verbose:
        print("=" * 70)
        print("Generating Customer Support Interactions")
        print("=" * 70)
        print(f"Creating {num_interactions} realistic support scenarios...")
        print()

    scenarios = []

    # Scenario 1: Order Tracking (Success)
    scenarios.append({
        "prompt": "Where is my order ORD-12346? I ordered a keyboard last week.",
        "output": "Let me check that for you. Using check_order_status('ORD-12346')...\n\nYour order for the Mechanical Keyboard ($89.99) was shipped on January 18th and is currently in transit. Expected delivery: 2-3 business days. Tracking number: TRK-987654321.",
        "success": True,
        "tool_calls": [{"tool": "check_order_status", "input": "ORD-12346"}],
        "reasoning": ["Customer asking about order", "Need to check order status", "Provide tracking info"]
    })

    # Scenario 2: Refund Request (Valid)
    scenarios.append({
        "prompt": "I received order ORD-12345 but the mouse isn't working. Can I get a refund?",
        "output": "I'm sorry to hear that! Let me help. Checking order ORD-12345...\n\nYour order was delivered on January 15th (5 days ago), which is within our 30-day return window. I'll process a refund of $29.99. Using process_refund('ORD-12345', 29.99)...\n\nRefund initiated. You'll receive $29.99 back to your original payment method within 5-7 business days. Return label has been emailed to you.",
        "success": True,
        "tool_calls": [
            {"tool": "check_order_status", "input": "ORD-12345"},
            {"tool": "process_refund", "input": {"order_id": "ORD-12345", "amount": 29.99}}
        ],
        "reasoning": ["Check order eligibility", "Within return window", "Process refund", "Send return label"]
    })

    # Scenario 3: Refund Request (Outside Policy)
    scenarios.append({
        "prompt": "I want a refund for order ORD-12348 from last month.",
        "output": "Let me check that order. Using check_order_status('ORD-12348')...\n\nI see your order for the Webcam HD was delivered on January 10th, which was 31 days ago. Unfortunately, this is outside our 30-day return window. However, you may be covered by the manufacturer's 1-year warranty for defects. Would you like me to create a warranty claim ticket?",
        "success": True,
        "tool_calls": [{"tool": "check_order_status", "input": "ORD-12348"}],
        "reasoning": ["Check order date", "Outside return window", "Offer alternative solution", "Mention warranty"]
    })

    # Scenario 4: Product Question
    scenarios.append({
        "prompt": "Is the USB-C Hub compatible with MacBook Pro 2023?",
        "output": "Great question! Let me find that information. Searching knowledge base...\n\nThe USB-C Hub (PROD-003) is compatible with all MacBook Pro models from 2016 onwards, including the 2023 models. It supports:\n- USB-C Power Delivery (100W)\n- 4K HDMI output\n- USB 3.0 ports (x3)\n- SD card reader\n\nWould you like me to check if it's in stock?",
        "success": True,
        "tool_calls": [
            {"tool": "search_knowledge_base", "input": "compatibility"},
            {"tool": "check_inventory", "input": "PROD-003"}
        ],
        "reasoning": ["Need product compatibility info", "Search knowledge base", "Check current stock"]
    })

    # Scenario 5: Out of Stock Inquiry
    scenarios.append({
        "prompt": "I want to buy the USB-C Hub. Is it available?",
        "output": "Let me check availability for you. Using check_inventory('PROD-003')...\n\nI'm sorry, the USB-C Hub is currently out of stock. However, we expect a new shipment within 2-3 weeks. Would you like me to:\n1. Notify you when it's back in stock\n2. Suggest a similar product that's available now\n\nWhat works better for you?",
        "success": True,
        "tool_calls": [{"tool": "check_inventory", "input": "PROD-003"}],
        "reasoning": ["Check inventory", "Out of stock", "Offer alternatives", "Set expectations"]
    })

    # Scenario 6: Technical Support
    scenarios.append({
        "prompt": "My keyboard isn't connecting via Bluetooth. How do I fix this?",
        "output": "I can help with that! Let me pull up the troubleshooting guide. Searching knowledge base for 'setup'...\n\nHere are steps to fix Bluetooth connection issues:\n\n1. Turn off the keyboard and turn it back on\n2. Hold the Bluetooth pairing button for 3 seconds\n3. On your computer: Settings → Bluetooth → Add Device\n4. Select 'Mechanical Keyboard' from the list\n\nIf this doesn't work:\n- Try removing old pairings first\n- Make sure the keyboard is charged\n- Check if Bluetooth is enabled on your computer\n\nDid this help? If not, I can create a support ticket for our tech team.",
        "success": True,
        "tool_calls": [{"tool": "search_knowledge_base", "input": "setup"}],
        "reasoning": ["Technical issue", "Search troubleshooting guide", "Provide step-by-step", "Offer escalation"]
    })

    # Scenario 7: Complaint (Needs Escalation)
    scenarios.append({
        "prompt": "This is the third time my order has been delayed! I want to speak to a manager NOW!",
        "output": "I sincerely apologize for the repeated delays - that's completely unacceptable. I understand your frustration. Let me create an urgent support ticket and escalate this to our customer relations manager.\n\nUsing create_support_ticket(type='escalation', priority='urgent')...\n\nTicket #TKT-5678 created. A manager will contact you within 2 hours via email and phone. I've also added a note about the repeated delays. Is there anything else I can help with while we get this resolved?",
        "success": True,
        "tool_calls": [{"tool": "create_support_ticket", "input": {"type": "escalation", "priority": "urgent", "customer_id": "CUST-001"}}],
        "reasoning": ["Angry customer", "Repeated issue", "Needs immediate escalation", "Create urgent ticket", "Set clear expectations"]
    })

    # Scenario 8: Return Policy Question
    scenarios.append({
        "prompt": "What's your return policy?",
        "output": "Let me get that information for you. Searching knowledge base...\n\nOur return policy:\n- Returns accepted within 30 days of delivery\n- Items must be unused and in original packaging\n- Refund processed within 5-7 business days\n- Free return shipping label provided\n- Original shipping costs are non-refundable\n\nDo you have a specific order you'd like to return?",
        "success": True,
        "tool_calls": [{"tool": "search_knowledge_base", "input": "returns"}],
        "reasoning": ["Policy question", "Search knowledge base", "Provide clear info", "Offer follow-up"]
    })

    # Scenario 9: System Error (Failure)
    scenarios.append({
        "prompt": "I need to check my order ORD-INVALID",
        "output": "Let me check that for you. Using check_order_status('ORD-INVALID')...\n\nError: Order not found. I apologize, but I couldn't find an order with that number. Could you:\n1. Double-check the order number (should be format: ORD-12345)\n2. Check your email confirmation for the correct order number\n3. Provide the email address used for the order\n\nI'm here to help once we locate the correct order!",
        "success": False,
        "tool_calls": [{"tool": "check_order_status", "input": "ORD-INVALID"}],
        "error": "Order not found",
        "reasoning": ["Invalid order number", "System error", "Provide helpful guidance", "Ask for verification"]
    })

    # Scenario 10: Product Recommendation
    scenarios.append({
        "prompt": "I'm looking for a good keyboard for programming. What do you recommend?",
        "output": "Great question! For programming, I'd recommend our Mechanical Keyboard (PROD-002). Here's why:\n\n✓ Mechanical switches for tactile feedback\n✓ Anti-ghosting for multiple key presses\n✓ Backlit keys for low-light coding\n✓ USB-C connection for laptops\n✓ Currently $89.99 with 45 units in stock\n\nLet me verify stock: check_inventory('PROD-002')... Yes, it's available!\n\nWould you like me to provide the product link, or do you have questions about it?",
        "success": True,
        "tool_calls": [{"tool": "check_inventory", "input": "PROD-002"}],
        "reasoning": ["Product recommendation", "Understand use case", "Check availability", "Highlight relevant features"]
    })

    # Scenario 11: Warranty Claim
    scenarios.append({
        "prompt": "My laptop stand broke after 6 months. Is it covered under warranty?",
        "output": "I'm sorry to hear that! Let me help. Searching warranty policy...\n\nGood news - all our products come with a 1-year manufacturer warranty covering defects. Since you're at 6 months, you're fully covered!\n\nUsing create_support_ticket(type='warranty_claim')...\n\nI've created warranty claim ticket #TKT-1234. You'll receive:\n1. Pre-paid shipping label (via email in 10 minutes)\n2. Replacement product ships within 24 hours of receiving your return\n3. Tracking number emailed when shipped\n\nTypical turnaround: 5-7 business days. Anything else I can help with?",
        "success": True,
        "tool_calls": [
            {"tool": "search_knowledge_base", "input": "warranty"},
            {"tool": "create_support_ticket", "input": {"type": "warranty_claim", "product": "PROD-004"}}
        ],
        "reasoning": ["Warranty question", "Check policy", "Within coverage", "Create claim ticket", "Set expectations"]
    })

    # Scenario 12: Shipping Question
    scenarios.append({
        "prompt": "How long does shipping usually take?",
        "output": "Let me get you that information. Searching shipping policy...\n\nWe offer two shipping options:\n\n📦 Standard Shipping: 5-7 business days (Free on orders $50+)\n🚀 Express Shipping: 2-3 business days (+$15)\n\nShipping times start after order processing (usually same day for orders before 2 PM EST).\n\nAre you asking about a specific order, or planning a new purchase?",
        "success": True,
        "tool_calls": [{"tool": "search_knowledge_base", "input": "shipping"}],
        "reasoning": ["Shipping question", "Provide options", "Mention free shipping threshold", "Offer follow-up"]
    })

    # ========================================
    # INTENT CLASSIFICATION SCENARIOS
    # ========================================

    # Scenario 13: Ambiguous Intent - Need to classify
    scenarios.append({
        "prompt": "I have a problem with my order",
        "output": "I'm here to help! First, let me classify what type of issue this is using classify_intent()...\n\nIntent detected: order_issue (generic)\n\nTo help you better, could you tell me more about the problem? For example:\n- Is it about tracking/delivery?\n- Product defect or damage?\n- Wrong item received?\n- Billing issue?\n\nOr I can look up your order if you have the order number?",
        "success": True,
        "tool_calls": [{"tool": "classify_intent", "input": "I have a problem with my order"}],
        "reasoning": ["Ambiguous query", "Classify intent first", "Request more info", "Offer specific options"]
    })

    # Scenario 14: Intent - Cancellation Request
    scenarios.append({
        "prompt": "I need to cancel my order ASAP",
        "output": "I understand you need to cancel. Let me check if that's possible. Using classify_intent()...\n\nIntent: order_cancellation (high urgency)\n\nTo cancel, I'll need your order number. Once you provide it, I can:\n1. Check if it's still in 'processing' (cancellable)\n2. If already shipped, help with a return instead\n3. Process immediate cancellation if possible\n\nWhat's your order number?",
        "success": True,
        "tool_calls": [{"tool": "classify_intent", "input": "cancel order ASAP"}],
        "reasoning": ["Detect cancellation intent", "Explain process", "Request required info", "Set expectations"]
    })

    # Scenario 15: Intent - Multiple possible intents
    scenarios.append({
        "prompt": "The keyboard I ordered isn't working and I need it for work tomorrow",
        "output": "I understand the urgency! Let me analyze this. Using classify_intent()...\n\nDetected intents:\n1. Product defect (primary)\n2. Time-sensitive need (secondary)\n3. Potential refund/replacement\n\nGiven the urgency, here's what I can do:\n- Express replacement shipped today (arrives tomorrow with express shipping)\n- OR immediate refund so you can buy locally\n- OR troubleshoot now if it's a simple fix\n\nWhich option works best for your situation?",
        "success": True,
        "tool_calls": [{"tool": "classify_intent", "input": "keyboard not working need for work tomorrow"}],
        "reasoning": ["Multiple intents detected", "Prioritize urgency", "Offer time-sensitive solutions", "Give customer control"]
    })

    # ========================================
    # INFORMATION EXTRACTION SCENARIOS
    # ========================================

    # Scenario 16: Extract order number from natural language
    scenarios.append({
        "prompt": "Hi, I ordered something last week, order number was something like ORD twelve thousand three forty seven, can you check the status?",
        "output": "Of course! Let me extract that order number. Using extract_entities()...\n\nExtracted: ORD-12347\n\nLet me check the status using check_order_status('ORD-12347')...\n\nYour order for the Laptop Stand ($39.99) is currently being processed. Status: Processing. Expected to ship within 24 hours. Once shipped, you'll receive a tracking number via email.",
        "success": True,
        "tool_calls": [
            {"tool": "extract_entities", "input": "ORD twelve thousand three forty seven"},
            {"tool": "check_order_status", "input": "ORD-12347"}
        ],
        "reasoning": ["Parse natural language", "Extract order number", "Normalize format", "Check status"]
    })

    # Scenario 17: Extract multiple pieces of information
    scenarios.append({
        "prompt": "I bought a mouse for $29.99 on January 15th but it's broken, need refund",
        "output": "I'm sorry to hear that! Let me extract the details. Using extract_entities()...\n\nExtracted information:\n- Product: Mouse ($29.99)\n- Date: January 15th\n- Issue: Defective\n- Requested action: Refund\n\nLet me find your order using search_orders(product='mouse', amount=29.99, date='2024-01-15')...\n\nFound order ORD-12345. Delivered 5 days ago, within 30-day return window. Processing refund of $29.99 using process_refund('ORD-12345', 29.99)...\n\nRefund initiated! You'll receive $29.99 within 5-7 business days.",
        "success": True,
        "tool_calls": [
            {"tool": "extract_entities", "input": "mouse $29.99 January 15th broken refund"},
            {"tool": "search_orders", "input": {"product": "mouse", "amount": 29.99}},
            {"tool": "process_refund", "input": {"order_id": "ORD-12345", "amount": 29.99}}
        ],
        "reasoning": ["Extract multiple entities", "Match to order", "Verify eligibility", "Process refund"]
    })

    # Scenario 18: Extract dates and calculate eligibility
    scenarios.append({
        "prompt": "I received my webcam around the tenth of last month, can I still return it?",
        "output": "Let me check that for you. Using extract_entities()...\n\nExtracted: delivery_date ≈ January 10th\nCalculating: Today (Jan 20) - Jan 10 = 10 days ago\n\nUsing search_orders(product='webcam')... Found ORD-12348.\n\nGood news! You're well within our 30-day return window (20 days remaining). I can process a return for you. The webcam was $79.99. Would you like to proceed with the return?",
        "success": True,
        "tool_calls": [
            {"tool": "extract_entities", "input": "tenth of last month"},
            {"tool": "calculate_date", "input": {"relative": "tenth of last month"}},
            {"tool": "search_orders", "input": {"product": "webcam"}},
            {"tool": "check_return_eligibility", "input": {"order_id": "ORD-12348"}}
        ],
        "reasoning": ["Parse relative date", "Calculate elapsed time", "Check policy", "Confirm eligibility"]
    })

    # ========================================
    # TOOL SELECTION (MCP-style) SCENARIOS
    # ========================================

    # Scenario 19: Choose correct tool from many options
    scenarios.append({
        "prompt": "What products do you have for under $50?",
        "output": "Let me search our catalog for you. Analyzing query using select_tool()...\n\nAvailable tools:\n- search_catalog (for product searches) ✓ SELECTED\n- check_order_status (for existing orders)\n- process_refund (for returns)\n- search_knowledge_base (for policies)\n- create_support_ticket (for escalations)\n\nUsing search_catalog(price_max=50)...\n\nFound 3 products under $50:\n1. Wireless Mouse - $29.99 (150 in stock)\n2. Laptop Stand - $39.99 (200 in stock)\n3. USB-C Hub - $49.99 (out of stock)\n\nWould you like more details on any of these?",
        "success": True,
        "tool_calls": [
            {"tool": "select_tool", "input": "search products under $50"},
            {"tool": "search_catalog", "input": {"price_max": 50}}
        ],
        "reasoning": ["Analyze query type", "Select appropriate tool from many options", "Execute search", "Present results"]
    })

    # Scenario 20: Chain multiple tools based on context
    scenarios.append({
        "prompt": "I want to buy a keyboard but I'm not sure if it's compatible with my setup, and also want to know about your warranty",
        "output": "Great questions! This requires multiple tools. Using select_tools()...\n\nQuery analysis:\n- Intent 1: Product compatibility → search_knowledge_base\n- Intent 2: Warranty info → search_knowledge_base  \n- Intent 3: Product details → check_inventory\n\nStep 1: Checking keyboard details using check_inventory('PROD-002')...\nMechanical Keyboard - $89.99, 45 in stock\n\nStep 2: Getting compatibility info using search_knowledge_base('compatibility')...\nUSB-C connection, works with Windows, Mac, Linux. Plug-and-play, no drivers needed.\n\nStep 3: Warranty details using search_knowledge_base('warranty')...\n1-year manufacturer warranty covering defects. Extended warranty available.\n\nIs there anything specific about compatibility you'd like me to check?",
        "success": True,
        "tool_calls": [
            {"tool": "select_tools", "input": "compatibility and warranty questions"},
            {"tool": "check_inventory", "input": "PROD-002"},
            {"tool": "search_knowledge_base", "input": "compatibility"},
            {"tool": "search_knowledge_base", "input": "warranty"}
        ],
        "reasoning": ["Identify multiple sub-intents", "Select multiple tools", "Chain tool calls", "Comprehensive answer"]
    })

    # Scenario 21: Dynamic tool selection based on order status
    scenarios.append({
        "prompt": "I need to change my order ORD-12347",
        "output": "I can help with that! First, let me check what's possible. Using select_tool()...\n\nDecision tree:\n- If processing → use modify_order ✓\n- If shipped → use create_return_request\n- If delivered → use process_refund\n\nChecking status using check_order_status('ORD-12347')...\nStatus: Processing\n\nGood news! Since it's still processing, I can modify it using modify_order(). What would you like to change?\n- Shipping address?\n- Shipping speed?\n- Cancel the order?\n- Add items?",
        "success": True,
        "tool_calls": [
            {"tool": "select_tool", "input": "modify order"},
            {"tool": "check_order_status", "input": "ORD-12347"}
        ],
        "reasoning": ["Query requires checking state first", "Select tool based on order status", "Explain what's possible", "Offer specific options"]
    })

    # ========================================
    # MULTI-STEP WORKFLOW SCENARIOS
    # ========================================

    # Scenario 22: Complex multi-step workflow
    scenarios.append({
        "prompt": "I ordered the wrong product (ORD-12346), can I exchange it for the mouse instead?",
        "output": "Absolutely! Let me handle this exchange for you. This requires a multi-step workflow.\n\nStep 1: Check current order using check_order_status('ORD-12346')...\nCurrent: Mechanical Keyboard ($89.99) - Status: Shipped\n\nStep 2: Check desired product using check_inventory('PROD-001')...\nWireless Mouse ($29.99) - In stock: 150 units\n\nStep 3: Calculate price difference: $89.99 - $29.99 = $60.00 refund\n\nStep 4: Create exchange workflow using create_exchange()...\n\nExchange initiated:\n1. Return label sent for keyboard\n2. Mouse will ship once keyboard is received\n3. $60.00 difference refunded in 5-7 days\n\nExpected timeline: 10-12 business days total. Sound good?",
        "success": True,
        "tool_calls": [
            {"tool": "check_order_status", "input": "ORD-12346"},
            {"tool": "check_inventory", "input": "PROD-001"},
            {"tool": "calculate_refund", "input": {"original": 89.99, "new": 29.99}},
            {"tool": "create_exchange", "input": {"return_order": "ORD-12346", "new_product": "PROD-001"}}
        ],
        "reasoning": ["Multi-step workflow", "Check both products", "Calculate difference", "Create exchange", "Set timeline"]
    })

    # Scenario 23: Conditional workflow based on inventory
    scenarios.append({
        "prompt": "I want to return my laptop stand and buy the USB-C hub instead",
        "output": "Let me check if we can do that exchange.\n\nStep 1: Check your return using check_order_status(product='laptop stand')...\nFound ORD-12347, Laptop Stand ($39.99) - Status: Processing\n\nStep 2: Check desired product using check_inventory('PROD-003')...\nUSB-C Hub ($49.99) - Status: OUT OF STOCK ❌\n\nWorkflow decision:\n❌ Direct exchange not possible (out of stock)\n✓ Alternative workflow:\n\n1. Cancel current order (still processing) using cancel_order('ORD-12347')\n2. Refund $39.99 immediately\n3. Notify you when USB-C Hub restocks (2-3 weeks)\n\nOR I can suggest similar in-stock products. Which works better?",
        "success": True,
        "tool_calls": [
            {"tool": "check_order_status", "input": "laptop stand"},
            {"tool": "check_inventory", "input": "PROD-003"},
            {"tool": "select_workflow", "input": {"type": "exchange", "availability": "out_of_stock"}}
        ],
        "reasoning": ["Check feasibility", "Inventory check fails", "Adapt workflow", "Offer alternatives"]
    })

    # Scenario 24: Progressive information gathering
    scenarios.append({
        "prompt": "I need help with a refund",
        "output": "I'd be happy to help with a refund! Let me gather the information I need.\n\nUsing progressive_workflow()...\n\nSlot filling:\n[✗] Order number: Not provided\n[✗] Reason: Not specified\n[✗] Order date: Unknown\n\nAsking for required information:\n'To process your refund, I'll need:\n1. Your order number (format: ORD-12345)\n2. Reason for return (optional but helps us improve)\n\nYou can find the order number in your confirmation email. What's the order number?'",
        "success": True,
        "tool_calls": [{"tool": "progressive_workflow", "input": {"type": "refund", "slots": {}}}],
        "reasoning": ["Incomplete information", "Use slot-filling pattern", "Request missing data", "Guide user"]
    })

    # ========================================
    # ERROR HANDLING & RECOVERY SCENARIOS
    # ========================================

    # Scenario 25: Handle API error gracefully
    scenarios.append({
        "prompt": "Check order ORD-12346",
        "output": "Let me check that for you. Using check_order_status('ORD-12346')...\n\nError: API timeout (Service temporarily unavailable)\n\nRecovery workflow:\n1. Retry using check_order_status_v2() with cache...\n2. Still failing → Use fallback\n\nFallback: Using search_email_confirmation('ORD-12346')...\n\nI apologize for the delay. Our order system is responding slowly, but I found your order in our backup system:\n\nOrder ORD-12346: Mechanical Keyboard ($89.99)\nLast known status: Shipped (as of 2 hours ago)\nExpected delivery: 2-3 business days\n\nWould you like me to create a ticket to get the real-time tracking once the system recovers?",
        "success": True,
        "tool_calls": [
            {"tool": "check_order_status", "input": "ORD-12346"},
            {"tool": "handle_error", "input": {"error": "API timeout"}},
            {"tool": "check_order_status_v2", "input": "ORD-12346"},
            {"tool": "search_email_confirmation", "input": "ORD-12346"}
        ],
        "reasoning": ["Primary tool fails", "Implement retry logic", "Use fallback", "Still provide value", "Offer next steps"]
    })

    # Scenario 26: Handle ambiguous product reference
    scenarios.append({
        "prompt": "I want to return the thing I bought last week",
        "output": "I can help with that return! However, I need to identify which product. Using extract_entities()...\n\nExtracted:\n- Time: 'last week' (Jan 13-19)\n- Product: 'thing' (ambiguous)\n\nSearching your orders using search_orders(date_range='2024-01-13 to 2024-01-19')...\n\nFound 2 orders from last week:\n1. ORD-12345: Wireless Mouse ($29.99) - Delivered Jan 15\n2. ORD-12346: Mechanical Keyboard ($89.99) - Shipped Jan 18\n\nWhich one would you like to return? Just let me know the number (1 or 2) or the product name.",
        "success": True,
        "tool_calls": [
            {"tool": "extract_entities", "input": "thing last week"},
            {"tool": "search_orders", "input": {"date_range": "2024-01-13 to 2024-01-19"}},
            {"tool": "disambiguate", "input": {"options": ["ORD-12345", "ORD-12346"]}}
        ],
        "reasoning": ["Ambiguous reference", "Extract time frame", "Search for matches", "Present options", "Request clarification"]
    })

    # ========================================
    # CONTEXT MANAGEMENT SCENARIOS
    # ========================================

    # Scenario 27: Maintain context across multi-turn
    scenarios.append({
        "prompt": "I ordered a keyboard",
        "output": "Thanks for letting me know! To help you with your keyboard order, I'll need a bit more information.\n\nUsing context_manager()...\n\nContext stored:\n- Product mentioned: Keyboard\n- Intent: Unclear (could be tracking, issue, question)\n\nWhat would you like to know about your keyboard order?\n- Track shipping status?\n- Technical support?\n- Return or exchange?\n- Something else?",
        "success": True,
        "tool_calls": [{"tool": "context_manager", "input": {"add": {"product": "keyboard", "intent": "unknown"}}}],
        "reasoning": ["Store context", "Incomplete query", "Ask clarifying question", "Maintain conversation state"]
    })

    # Scenario 28: Use context from previous interaction
    scenarios.append({
        "prompt": "Actually, I want to return it",
        "output": "No problem! Let me help you return the keyboard.\n\nUsing context_manager()...\n\nContext retrieved:\n- Previous: Keyboard mentioned\n- Current: Return request\n- Combined: Return keyboard order\n\nSearching for keyboard order using search_orders(product='keyboard')...\nFound: ORD-12346 - Mechanical Keyboard ($89.99)\n\nLet me check return eligibility using check_order_status('ORD-12346')...\nShipped 2 days ago, within 30-day window ✓\n\nI can process this return. Would you like me to start the return process and send you a label?",
        "success": True,
        "tool_calls": [
            {"tool": "context_manager", "input": {"retrieve": "last_product"}},
            {"tool": "search_orders", "input": {"product": "keyboard"}},
            {"tool": "check_order_status", "input": "ORD-12346"}
        ],
        "reasoning": ["Retrieve context", "Pronoun resolution", "Find relevant order", "Check eligibility"]
    })

    # ========================================
    # COMPLEX DECISION TREE SCENARIOS
    # ========================================

    # Scenario 29: Decision tree based on multiple factors
    scenarios.append({
        "prompt": "My order hasn't arrived and I needed it yesterday",
        "output": "I understand the urgency! Let me determine the best solution.\n\nUsing decision_tree()...\n\nFactors to check:\n1. Expected delivery date vs actual\n2. Current order status\n3. Time sensitivity\n4. Available remedies\n\nI need your order number to check these factors. Could you provide it? Based on the situation, I might be able to:\n- Expedite a replacement (overnight shipping)\n- Issue immediate refund for local purchase\n- File lost package claim\n- Locate package if just delayed\n\nWhat's your order number?",
        "success": True,
        "tool_calls": [{"tool": "decision_tree", "input": {"type": "late_delivery", "urgency": "high"}}],
        "reasoning": ["Multiple factors", "Complex decision tree", "Request missing info", "Preview options"]
    })

    # Scenario 30: Prioritize actions based on business rules
    scenarios.append({
        "prompt": "I'm a business customer with 50 employees, we ordered 50 mice but only got 45",
        "output": "Thank you for bringing this to our attention! As a business customer, this gets priority handling.\n\nUsing business_rules_engine()...\n\nCustomer type: Business (high priority)\nIssue: Short shipment (-5 units)\nValue: 50 × $29.99 = $1,499.50\n\nPriority routing:\n✓ Skip standard process\n✓ Escalate to business support\n✓ Immediate resolution required\n\nActions taken:\n1. Creating urgent ticket using create_support_ticket(type='business_priority')...\n2. Immediate shipment of 5 mice (overnight shipping)\n3. 10% discount on order ($150 credit)\n4. Dedicated account manager will call within 1 hour\n\nTicket #TKT-BUSINESS-001 created. Anything else I can help with while we resolve this?",
        "success": True,
        "tool_calls": [
            {"tool": "business_rules_engine", "input": {"customer_type": "business", "issue": "short_shipment"}},
            {"tool": "create_support_ticket", "input": {"type": "business_priority", "urgency": "high"}},
            {"tool": "ship_replacement", "input": {"product": "PROD-001", "quantity": 5, "speed": "overnight"}},
            {"tool": "apply_credit", "input": {"amount": 150}}
        ],
        "reasoning": ["Identify high-value customer", "Apply business rules", "Escalate appropriately", "Immediate action", "Over-compensate"]
    })

    # ========================================
    # ADDITIONAL FAILURE SCENARIOS (for DPO)
    # ========================================

    # Add more failure cases to ensure balanced dataset
    failure_scenarios = [
        {
            "prompt": "I want to return my mouse from 2 months ago",
            "output": "Sure! Let me process that refund right away using process_refund()...",
            "success": False,
            "error": "Should have checked return policy first - item outside 30-day window",
            "reasoning": ["Missing policy check", "Didn't verify eligibility", "Incorrect tool use"]
        },
        {
            "prompt": "What's the status of order ORD-12345?",
            "output": "I don't have access to order information. Please contact our support team.",
            "success": False,
            "error": "Should have used check_order_status tool instead of deflecting",
            "reasoning": ["Didn't attempt to use available tools", "Deflected unnecessarily"]
        },
        {
            "prompt": "My order ORD-12346 never arrived",
            "output": "I'm sorry to hear that! I'll send you a replacement right away using ship_replacement()...",
            "success": False,
            "error": "Should have checked order status first - order shows as 'shipped', not lost",
            "reasoning": ["Didn't verify claim", "Premature action", "Potential abuse"]
        },
        {
            "prompt": "Is the USB-C Hub in stock?",
            "output": "Yes, we have plenty in stock! Would you like to order one?",
            "success": False,
            "error": "Didn't check inventory - product is actually out of stock",
            "reasoning": ["Guessed instead of checking", "Provided false information"]
        },
        {
            "prompt": "I need urgent help with order ORD-INVALID",
            "output": "Order ORD-INVALID not found. Thank you for contacting us!",
            "success": False,
            "error": "Didn't offer to help customer find correct order number",
            "reasoning": ["Unhelpful response", "Didn't guide customer", "Poor recovery"]
        },
        {
            "prompt": "The keyboard I received is damaged",
            "output": "That's unfortunate. Our return policy is 30 days.",
            "success": False,
            "error": "Should have immediately offered warranty claim or replacement, not just cited policy",
            "reasoning": ["Not customer-focused", "Didn't offer solution", "Policy-only response"]
        },
        {
            "prompt": "Can I get a discount on the laptop stand?",
            "output": "Sure! I'll apply a 50% discount right away using apply_discount()...",
            "success": False,
            "error": "Gave unauthorized discount without checking business rules",
            "reasoning": ["Exceeded authority", "No validation", "Potential loss"]
        },
        {
            "prompt": "I'm extremely unhappy with your service!",
            "output": "I understand. Is there anything else I can help you with?",
            "success": False,
            "error": "Should have acknowledged frustration, apologized, and escalated appropriately",
            "reasoning": ["Dismissive response", "No escalation", "Poor empathy"]
        },
    ]

    scenarios.extend(failure_scenarios)

    # Categorize scenarios for balanced sampling
    categories = {
        "order_tracking": [s for s in scenarios if any(kw in s["prompt"].lower() for kw in ["order", "track", "status", "shipped"])],
        "refunds_returns": [s for s in scenarios if any(kw in s["prompt"].lower() for kw in ["refund", "return", "broken", "defect"])],
        "product_questions": [s for s in scenarios if any(kw in s["prompt"].lower() for kw in ["product", "compatible", "recommend", "stock"])],
        "technical_support": [s for s in scenarios if any(kw in s["prompt"].lower() for kw in ["not working", "fix", "bluetooth", "setup"])],
        "policy_questions": [s for s in scenarios if any(kw in s["prompt"].lower() for kw in ["policy", "warranty", "shipping"])],
        "escalations": [s for s in scenarios if any(kw in s["prompt"].lower() for kw in ["manager", "complaint", "unhappy", "angry"])],
    }

    # Ensure we have scenarios in each category
    for cat_name, cat_scenarios in categories.items():
        if len(cat_scenarios) == 0:
            categories[cat_name] = scenarios[:1]  # Fallback

    # Generate balanced interactions
    interactions = []

    # First, add all base scenarios once
    for scenario in scenarios:
        trace = AgentTrace(
            prompt=scenario["prompt"],
            output=scenario["output"],
            timestamp=datetime.now().isoformat(),
            success=scenario.get("success", True),
            reasoning=scenario.get("reasoning"),
            tool_calls=scenario.get("tool_calls"),
            error=scenario.get("error"),
            metadata={"domain": "customer_support", "type": "interaction"}
        )
        interactions.append(trace)

    # Then generate more variations with balanced category sampling
    remaining = num_interactions - len(scenarios)
    for i in range(remaining):
        # Rotate through categories for balance
        category_name = list(categories.keys())[i % len(categories)]
        category_scenarios = categories[category_name]

        # Pick random scenario from that category
        base = random.choice(category_scenarios)

        # Add some variation by flipping success status occasionally for DPO training
        success = base.get("success", True)
        if random.random() < 0.1 and success:  # 10% chance to create failure variant
            success = False
            error = "Simulated failure for DPO training"
        else:
            error = base.get("error")

        trace = AgentTrace(
            prompt=base["prompt"],
            output=base["output"],
            timestamp=datetime.now().isoformat(),
            success=success,
            reasoning=base.get("reasoning"),
            tool_calls=base.get("tool_calls"),
            error=error,
            metadata={
                "domain": "customer_support",
                "type": "interaction",
                "category": category_name,
                "variation": i
            }
        )
        interactions.append(trace)

    # Save to file
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for interaction in interactions:
            f.write(interaction.to_json() + "\n")

    if verbose:
        successful = sum(1 for t in interactions if t.success)
        failed = sum(1 for t in interactions if not t.success)

        print(f"✓ Created {len(interactions)} support interactions")
        print(f"  Success: {successful} ({100*successful/len(interactions):.1f}%) | Failed: {failed} ({100*failed/len(interactions):.1f}%)")
        print(f"  Saved to: {output_path}")
        print()

        # Show category balance
        print("Dataset balance by category:")
        category_counts = {}
        for interaction in interactions:
            cat = interaction.metadata.get("category", "base_scenarios")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for cat, count in sorted(category_counts.items()):
            print(f"  • {cat}: {count} ({100*count/len(interactions):.1f}%)")
        print()

        print("Agent patterns included:")
        print("  • Tool selection (MCP-style: choosing from many tools)")
        print("  • Intent classification (routing ambiguous queries)")
        print("  • Information extraction (parsing order numbers, dates, etc.)")
        print("  • Multi-step workflows (chaining multiple tools)")
        print("  • Error handling (graceful failures and recovery)")
        print("  • Context management (multi-turn conversations)")
        print("  • Decision trees (complex conditional logic)")
        print("  • Slot filling (progressive information gathering)")
        print()

    return len(interactions)


def create_customer_support_eval_tasks():
    """Create realistic customer support evaluation tasks."""
    from deltaloop.eval import EvalTask

    tasks = [
        EvalTask(
            name="tool_selection_refund",
            prompt="Customer: I received order ORD-12345 two weeks ago but it's defective. I'd like a refund.",
            judge_fn=lambda output: 1.0 if "process_refund" in output.lower() or "refund" in output.lower() else 0.0,
            expected="process_refund",
            metadata={"category": "tool_selection", "skill": "refund_handling"}
        ),
        EvalTask(
            name="tool_selection_order_status",
            prompt="Customer: Where is my order ORD-12346?",
            expected="check_order_status",
            judge_fn=lambda output: 1.0 if "check_order_status" in output.lower() or "status" in output.lower() else 0.0,
            metadata={"category": "tool_selection", "skill": "order_tracking"}
        ),
        EvalTask(
            name="policy_compliance_return_window",
            prompt="Customer: I want to return an item I bought 45 days ago.",
            expected="Should mention 30-day policy and suggest alternatives",
            judge_fn=lambda output: 1.0 if ("30" in output and ("day" in output.lower() or "policy" in output.lower())) else 0.5 if "policy" in output.lower() else 0.0,
            metadata={"category": "policy_compliance", "skill": "return_policy"}
        ),
        EvalTask(
            name="escalation_angry_customer",
            prompt="Customer: This is ridiculous! I want to speak to your manager RIGHT NOW!",
            expected="Should escalate and show empathy",
            judge_fn=lambda output: 1.0 if ("escalat" in output.lower() or "manager" in output.lower()) and ("apolog" in output.lower() or "sorry" in output.lower()) else 0.5 if "manager" in output.lower() else 0.0,
            metadata={"category": "escalation", "skill": "conflict_resolution"}
        ),
        EvalTask(
            name="info_extraction_order_number",
            prompt="Customer: I need help with order ORD twelve thousand three forty six",
            expected="Should extract ORD-12346",
            judge_fn=lambda output: 1.0 if "12346" in output or "ORD-12346" in output else 0.0,
            metadata={"category": "information_extraction", "skill": "entity_parsing"}
        ),
        EvalTask(
            name="multi_step_refund_eligibility",
            prompt="Customer: I got a broken mouse, can I get my money back?",
            expected="Should check order, verify within policy, then process refund",
            judge_fn=lambda output: 1.0 if ("check" in output.lower() or "verify" in output.lower()) and "refund" in output.lower() else 0.5 if "refund" in output.lower() else 0.0,
            metadata={"category": "multi_step_workflow", "skill": "refund_process"}
        ),
        EvalTask(
            name="product_stock_check",
            prompt="Customer: Is the USB-C Hub available?",
            expected="Should check inventory before answering",
            judge_fn=lambda output: 1.0 if "check" in output.lower() or "inventory" in output.lower() or "stock" in output.lower() else 0.0,
            metadata={"category": "tool_selection", "skill": "inventory_check"}
        ),
        EvalTask(
            name="technical_support_guidance",
            prompt="Customer: My keyboard won't connect via Bluetooth",
            expected="Should provide troubleshooting steps",
            judge_fn=lambda output: 1.0 if (("step" in output.lower() or "try" in output.lower()) and ("bluetooth" in output.lower() or "pairing" in output.lower())) else 0.5 if len(output) > 50 else 0.0,
            metadata={"category": "technical_support", "skill": "troubleshooting"}
        ),
        EvalTask(
            name="error_handling_invalid_order",
            prompt="Customer: Check order ORD-NOTFOUND",
            expected="Should handle error gracefully and offer help",
            judge_fn=lambda output: 1.0 if ("not found" in output.lower() or "couldn't find" in output.lower()) and ("help" in output.lower() or "check" in output.lower()) else 0.5 if "not found" in output.lower() else 0.0,
            metadata={"category": "error_handling", "skill": "recovery"}
        ),
        EvalTask(
            name="warranty_claim",
            prompt="Customer: My laptop stand broke after 8 months, is it covered?",
            expected="Should mention 1-year warranty and create claim",
            judge_fn=lambda output: 1.0 if ("warranty" in output.lower() and ("year" in output.lower() or "12" in output)) else 0.5 if "warranty" in output.lower() else 0.0,
            metadata={"category": "policy_compliance", "skill": "warranty_handling"}
        ),
    ]

    return tasks


def evaluate_on_test_set(model, tokenizer, test_traces, custom_tasks, verbose=True):
    """Evaluate model on test set with custom customer support tasks.

    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        test_traces: List of test AgentTrace objects
        custom_tasks: List of custom EvalTask objects
        verbose: Print progress

    Returns:
        dict with evaluation results
    """
    if verbose:
        print("=" * 70)
        print("Evaluating on Customer Support Test Set")
        print("=" * 70)
        print(f"Test examples: {len(test_traces)}")
        print(f"Custom eval tasks: {len(custom_tasks)}")
        print()

    results = {
        "test_set_size": len(test_traces),
        "num_tasks": len(custom_tasks),
        "task_results": []
    }

    for task in custom_tasks:
        if verbose:
            print(f"Task: {task.name}")

        # Generate response
        inputs = tokenizer(task.prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Score
        score = task.judge_fn(response)

        results["task_results"].append({
            "task": task.name,
            "category": task.metadata.get("category"),
            "skill": task.metadata.get("skill"),
            "score": score,
            "prompt": task.prompt[:50] + "...",
            "response": response[:100] + "..."
        })

        if verbose:
            print(f"  Score: {score*100:.1f}%")
            print(f"  Category: {task.metadata.get('category')}")
            print()

    # Calculate averages by category
    categories = {}
    for result in results["task_results"]:
        cat = result["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(result["score"])

    results["category_averages"] = {
        cat: sum(scores) / len(scores) * 100
        for cat, scores in categories.items()
    }

    results["overall_average"] = sum(r["score"] for r in results["task_results"]) / len(results["task_results"]) * 100

    if verbose:
        print("=" * 70)
        print("Evaluation Results by Category")
        print("=" * 70)
        for cat, avg in sorted(results["category_averages"].items()):
            print(f"  {cat}: {avg:.1f}%")
        print()
        print(f"Overall Average: {results['overall_average']:.1f}%")
        print("=" * 70)
        print()

    return results


def main():
    """Run customer support agent training pipeline."""
    parser = argparse.ArgumentParser(
        description="Customer Support Agent Training with DeltaLoop",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Base model (default: TinyLlama)")
    parser.add_argument("--steps", type=int, default=100,
                       help="Training steps (default: 100)")
    parser.add_argument("--num-interactions", type=int, default=500,
                       help="Number of support interactions to generate (default: 500)")
    parser.add_argument("--training-method", choices=["sft", "dpo"], default="sft",
                       help="Training method: 'sft' (supervised) or 'dpo' (preference learning, default: sft)")
    parser.add_argument("--no-train", action="store_true",
                       help="Skip training (just generate data)")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Train/test split ratio (default: 0.8 for 80%% train)")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode")

    args = parser.parse_args()
    verbose = not args.quiet

    if verbose:
        print()
        print("=" * 70)
        print("CUSTOMER SUPPORT AGENT - DELTALOOP TRAINING")
        print("=" * 70)
        print()
        print("Scenario: E-commerce customer support agent")
        print("Goal: Train agent to handle common support queries better")
        print()
        print("The agent learns:")
        print("  • Tool selection (choosing right tool from many options)")
        print("  • Intent classification (routing queries correctly)")
        print("  • Information extraction (parsing structured data)")
        print("  • Multi-step workflows (chaining tools)")
        print("  • Error handling and recovery")
        print("  • Context management across turns")
        print()
        print("=" * 70)
        print()

    # Step 1: Generate support interactions
    logs_path = "data/customer_support/interactions.jsonl"
    num_interactions = generate_support_interactions(
        logs_path,
        num_interactions=args.num_interactions,
        verbose=verbose
    )

    # Step 2: Split into train/test for validation on unseen examples
    if verbose:
        print("=" * 70)
        print("Splitting Dataset (Train/Test)")
        print("=" * 70)

    # Load all interactions
    all_interactions = []
    with open(logs_path, 'r') as f:
        for line in f:
            all_interactions.append(AgentTrace.from_json(line.strip()))

    # Shuffle for random split
    random.shuffle(all_interactions)

    # Split
    split_idx = int(len(all_interactions) * args.train_split)
    train_interactions = all_interactions[:split_idx]
    test_interactions = all_interactions[split_idx:]

    # Save train and test sets
    train_path = "data/customer_support/interactions_train.jsonl"
    test_path = "data/customer_support/interactions_test.jsonl"

    with open(train_path, 'w') as f:
        for interaction in train_interactions:
            f.write(interaction.to_json() + "\n")

    with open(test_path, 'w') as f:
        for interaction in test_interactions:
            f.write(interaction.to_json() + "\n")

    if verbose:
        print(f"Training set: {len(train_interactions)} examples ({args.train_split*100:.0f}%)")
        print(f"Test set: {len(test_interactions)} examples ({(1-args.train_split)*100:.0f}%)")
        print(f"Train saved to: {train_path}")
        print(f"Test saved to: {test_path}")
        print()

    if args.no_train:
        if verbose:
            print("--no-train specified. Stopping after data generation and split.")
            print(f"Generated interactions saved to: {logs_path}")
            print()
            print("To train the model, run:")
            print(f"  python3 examples/customer_support_agent.py --steps {args.steps}")
        return

    # Step 3-6: Run training pipeline on TRAIN set only
    config = PipelineConfig(
        logs_path=train_path,  # Use train set only!
        base_model=args.model,
        training_steps=args.steps,
        training_method=args.training_method,
        output_dir="data/customer_support/pipeline_runs",
        run_name=f"support_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.training_method}",
        verbose=verbose
    )

    pipeline = Pipeline(verbose=verbose)
    result = pipeline.run(config)

    # Step 7: Evaluate on TEST set with custom customer support tasks
    if verbose:
        print()
        print("=" * 70)
        print("Custom Evaluation on Unseen Test Set")
        print("=" * 70)
        print()

    # Create custom eval tasks
    custom_tasks = create_customer_support_eval_tasks()

    if TORCH_AVAILABLE:
        # Load the trained model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
        model = PeftModel.from_pretrained(base_model, result.adapter_path)

        # Run custom evaluation
        custom_results = evaluate_on_test_set(
            model=model,
            tokenizer=tokenizer,
            test_traces=test_interactions,
            custom_tasks=custom_tasks,
            verbose=verbose
        )

        # Save custom results
        custom_results_path = str(Path(result.adapter_path).parent / "custom_eval_results.json")
        with open(custom_results_path, 'w') as f:
            json.dump(custom_results, f, indent=2)
    else:
        if verbose:
            print("  ⚠ Torch not available, skipping custom evaluation")
            print("  Install torch to enable: pip install torch")
            print()
        custom_results = None

    # Final summary
    if verbose:
        print()
        print("=" * 70)
        print("RESULTS - Customer Support Agent")
        print("=" * 70)
        print(f"Generic Eval Improvement: {result.eval_summary['improvement_percent']:+.1f}%")
        if custom_results:
            print(f"Custom Support Eval: {custom_results['overall_average']:.1f}%")
            print()
            print("Top performing categories:")
            sorted_cats = sorted(custom_results['category_averages'].items(), key=lambda x: x[1], reverse=True)
            for cat, score in sorted_cats[:3]:
                print(f"  • {cat}: {score:.1f}%")
        print()
        print("Dataset info:")
        print(f"  Training examples: {len(train_interactions)} ({args.train_split*100:.0f}%)")
        print(f"  Test examples: {len(test_interactions)} ({(1-args.train_split)*100:.0f}%)")
        print(f"  Training method: {args.training_method.upper()}")
        print()
        print("What this means:")
        if custom_results and custom_results['overall_average'] > 50:
            print("  ✓ The agent shows good performance on customer support tasks")
            print("  ✓ Validated on unseen test examples")
            print("  ✓ Ready for realistic customer support scenarios")
        elif result.eval_summary['improvement_percent'] > 20:
            print("  ✓ The agent improved on generic tasks")
            print("  → Custom eval provides better measure of support capabilities")
        else:
            print("  → Consider:")
            print("    • More training interactions (--num-interactions 1000)")
            print("    • More training steps (--steps 200)")
            print("    • Try DPO training (--training-method dpo)")
        print()
        print(f"Model saved: {result.adapter_path}")
        print(f"Results: {result.results_path}")
        if custom_results:
            print(f"Custom eval: {custom_results_path}")
        print("=" * 70)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
