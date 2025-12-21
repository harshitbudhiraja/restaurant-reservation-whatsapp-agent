import json
import os
import sys
import re
import inspect
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.booking_utils import book_table as actual_book_table
from recommendation_system.rs import get_recommendation as actual_get_recommendation
from agents.location_detector import load_locations

LOCATION_FILE = os.path.join(os.path.dirname(__file__), "../location.json")

def get_location_context():
    try:
        with open(LOCATION_FILE, "r", encoding="utf-8") as f:
            locations = json.load(f)
        info = []
        for loc_id, loc_data in locations.items():
            info.append(
                f"ID: {loc_data['id']}, Name: {loc_data['name']}, "
                f"Address: {loc_data['address']}, "
                f"Coordinates: ({loc_data['lat']}, {loc_data['long']}), "
                f"Capacity: {loc_data['total_capacity']}"
            )
        return "\n".join(info)
    except Exception as e:
        print("Could not load location data:", e)
        return "Location data not available"

def book_table(user_id: str, date: str, time: str, restaurant_location: str, number_of_people: int):
    try:
        if actual_book_table:
            result = actual_book_table(
                user_id=user_id,
                date=date,
                time=time,
                restaurant_location=restaurant_location,
                number_of_people=number_of_people,
            )
            return {
                "status": "success",
                "message": result,
                "location": restaurant_location,
                "date": date,
                "time": time,
                "guests": number_of_people,
            }
        else:
            return {
                "status": "success",
                "message": (
                    "Table booked successfully!\n"
                    "Location: Connaught Place\n"
                    "Date: 2024-06-14\n"
                    "Time: 7:00 PM\n"
                    f"Guests: {number_of_people} people"
                ),
                "location": "Connaught Place",
                "date": "2024-06-14",
                "time": "7:00 PM",
                "guests": 4,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error booking table: " + str(e),
            "location": restaurant_location,
            "date": date,
            "time": time,
            "guests": number_of_people,
        }

def get_recommendation(user_lat: float, user_long: float):
    try:
        if actual_get_recommendation:
            result, lat, long = actual_get_recommendation(user_lat, user_long)
            return {
                "status": "success",
                "message": result,
                "user_location": (user_lat, user_long),
                "count": 3
            }
        else:
            return {
                "status": "success",
                "message": "Top 3 Nearby Venues:\n"
                           "1. Connaught Place, New Delhi (0.5 km)\n"
                           "2. Khan Market, New Delhi (2.1 km)\n"
                           "3. Cyber City, Gurugram (15.3 km)",
                "user_location": (user_lat, user_long),
                "count": 3
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error getting recommendations: " + str(e),
            "user_location": (user_lat, user_long),
            "count": 0
        }

def get_available_locations():
    try:
        if load_locations:
            locations = load_locations()
            return {
                "status": "success",
                "locations": locations,
                "count": len(locations)
            }
        else:
            with open(LOCATION_FILE, "r", encoding="utf-8") as f:
                locations = json.load(f)
            return {
                "status": "success",
                "locations": list(locations.values()),
                "count": len(locations)
            }
    except Exception as e:
        return {
            "status": "error",
            "message": "Error loading locations: " + str(e),
            "locations": [],
            "count": 0
        }

AVAILABLE_FUNCTIONS = [
    book_table,
    get_recommendation,
    get_available_locations
]

def check_missing_parameters(function_name: str, provided_args: Dict[str, Any]) -> List[str]:
    func_obj = None
    for func in AVAILABLE_FUNCTIONS:
        if func.__name__ == function_name:
            func_obj = func
            break
    if func_obj is None:
        return []
    sig = inspect.signature(func_obj)
    required = []
    for pname, param in sig.parameters.items():
        if param.default == inspect.Parameter.empty:
            required.append(pname)
    missing = [p for p in required if p not in provided_args or provided_args[p] is None]
    return missing

def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    tool_calls = []

    # Try 1: JSON
    json_pattern = r'\{[^{}]*"function"[^{}]*"arguments"[^{}]*\}'
    for match in re.finditer(json_pattern, response, re.DOTALL):
        try:
            parsed = json.loads(match.group(0))
            if "function" in parsed and "arguments" in parsed:
                tool_calls.append({
                    "function": parsed["function"],
                    "arguments": parsed["arguments"],
                    "raw": match.group(0)
                })
        except Exception:
            continue

    # Try 2: <start_function_call> ... <end_function_call>
    start_token = "<start_function_call>"
    end_token = "<end_function_call>"
    if start_token in response and end_token in response:
        pat = f"{re.escape(start_token)}(.*?){re.escape(end_token)}"
        for m in re.finditer(pat, response, re.DOTALL):
            call_txt = m.group(1).strip()
            pc = _parse_function_call_text(call_txt)
            if pc:
                pc["raw"] = m.group(0)
                tool_calls.append(pc)

    # Try 3: call:function_name{...}
    call_pattern = r'call:(\w+)\{([^}]*)\}'
    for m in re.finditer(call_pattern, response):
        fname = m.group(1)
        arg_txt = m.group(2)
        args = _parse_arguments_text(arg_txt)
        tool_calls.append({
            "function": fname,
            "arguments": args,
            "raw": m.group(0)
        })

    # Try 4: function signature style
    for func in AVAILABLE_FUNCTIONS:
        fname = func.__name__
        pat = rf'\b{fname}\s*[\(\[]\s*([^\)\]]+)\s*[\)\]]'
        for m in re.finditer(pat, response, re.IGNORECASE):
            args = _parse_arguments_text(m.group(1))
            tool_calls.append({
                "function": fname,
                "arguments": args,
                "raw": m.group(0)
            })
    return tool_calls

def _parse_function_call_text(text: str) -> Optional[Dict[str, Any]]:
    m = re.match(r'call:(\w+)\{([^}]*)\}', text)
    if m:
        fname = m.group(1)
        args = _parse_arguments_text(m.group(2))
        return {
            "function": fname,
            "arguments": args
        }
    try:
        parsed = json.loads(text)
        if "function" in parsed:
            return {
                "function": parsed["function"],
                "arguments": parsed.get("arguments", {})
            }
    except Exception:
        return None
    return None

def _parse_arguments_text(args_text: str) -> Dict[str, Any]:
    args_dict = {}
    if not args_text.strip():
        return args_dict
    try:
        parsed = json.loads("{" + args_text + "}")
        return parsed
    except Exception:
        pass

    args_text = re.sub(r'<escape>(.*?)<escape>', r'"\1"', args_text)
    pat = r'(\w+):\s*([^,}]+)'
    for m in re.finditer(pat, args_text):
        key = m.group(1)
        value = m.group(2).strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        try:
            if value.isdigit():
                value = int(value)
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
        except Exception:
            pass
        args_dict[key] = value
    return args_dict

def analyze_query_parameters(user_query: str, function_name: str) -> Dict[str, Any]:
    func_obj = None
    for func in AVAILABLE_FUNCTIONS:
        if func.__name__ == function_name:
            func_obj = func
            break
    if func_obj is None:
        return {
            "function": function_name,
            "provided_params": {},
            "missing_params": [],
            "analysis": f"Function '{function_name}' not found"
        }
    sig = inspect.signature(func_obj)
    provided_params = {}

    people_match = re.search(r'(\d+)\s*(?:people|guests|persons|pax)', user_query, re.IGNORECASE)
    if people_match:
        provided_params['number_of_people'] = int(people_match.group(1))

    date_patterns = [
        r'tomorrow', r'today', r'\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}',
        r'\w+\s+\d{1,2}', r'\d{4}-\d{2}-\d{2}'
    ]
    for pattern in date_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            provided_params['date'] = match.group(0)
            break

    time_patterns = [
        r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', r'\d{1,2}\s*(?:AM|PM|am|pm)',
        r'\d{1,2}:\d{2}', r'(?:morning|afternoon|evening|night)'
    ]
    for pattern in time_patterns:
        match = re.search(pattern, user_query, re.IGNORECASE)
        if match:
            provided_params['time'] = match.group(0)
            break

    loc_match = re.search(r'at\s+([A-Z][a-zA-Z\s]+)', user_query)
    if loc_match:
        provided_params['restaurant_location'] = loc_match.group(1).strip()

    missing = check_missing_parameters(function_name, provided_params)
    text = f"Function: {function_name}\nFound {len(provided_params)} parameter(s): {list(provided_params.keys())}\nMissing {len(missing)} parameter(s): {missing}"
    return {
        "function": function_name,
        "provided_params": provided_params,
        "missing_params": missing,
        "analysis": text
    }

def generate_with_functiongemma(
    user_message: str,
    tokenizer,
    model,
    functions=None,
    include_location_context=False,
    max_new_tokens=256,
    do_sample=False,
    extract_tool_calls_flag=False
):
    if functions is None:
        functions = AVAILABLE_FUNCTIONS

    messages = [{"role": "user", "content": user_message}]
    if include_location_context:
        location_context = get_location_context()
        messages[0]["content"] += f"\n\nAvailable Locations:\n{location_context}"

    print("messages:", messages)
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=functions,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    if hasattr(model, "device"):
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    if extract_tool_calls_flag:
        tool_calls = extract_tool_calls(response)
        return {
            "response": response,
            "tool_calls": tool_calls
        }

    return response

if __name__ == "__main__":
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it")

    test_message = "I want to book a table for 4 people tomorrow at 7pm"

    print("User Query:", test_message)
    print("="*80)

    print("\n1. Checking missing parameters for 'book_table':")
    analysis = analyze_query_parameters(test_message, "book_table")
    print(analysis['analysis'])
    print("   Provided:", analysis['provided_params'])
    print("   Missing:", analysis['missing_params'])

    print("\n2. Generate response and extract tool calls:")
    result = generate_with_functiongemma(
        user_message=test_message,
        tokenizer=tokenizer,
        model=model,
        include_location_context=False,
        max_new_tokens=256,
        do_sample=False,
        extract_tool_calls_flag=True
    )
    print("\nRaw Response:", result['response'])
    print("\nExtracted Tool Calls:", len(result['tool_calls']), "found")

    for i, tool_call in enumerate(result['tool_calls'], 1):
        print(f"\nTool Call {i}:")
        print("  Function:", tool_call['function'])
        print("  Arguments:", tool_call['arguments'])
        missing = check_missing_parameters(tool_call['function'], tool_call['arguments'])
        if missing:
            print("  Missing parameters:", missing)
        else:
            print("  All required parameters provided")

    print("="*80)
