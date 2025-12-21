import json
import os
import sys
import re
import inspect
from typing import Dict, List, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import date

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.booking_utils import book_table as actual_book_table
from recommendation_system.rs import get_recommendation as actual_get_recommendation
from agents.location_detector import load_locations
LOCATION_FILE = os.path.join(os.path.dirname(__file__), "../location.json")


def get_location_context():
    """Loads all location data to include in context for large context testing"""
    try:
        with open(LOCATION_FILE, "r", encoding="utf-8") as f:
            locations = json.load(f)
        location_list = []
        for loc_id, loc_data in locations.items():
            location_list.append(
                f"ID: {loc_data['id']}, Name: {loc_data['name']}, "
                f"Address: {loc_data['address']}, "
                f"Coordinates: ({loc_data['lat']}, {loc_data['long']}), "
                f"Capacity: {loc_data['total_capacity']}"
            )
        return "\n".join(location_list)
    except Exception as e:
        print(f"Warning: Could not load location data: {e}")
        return "Location data not available"

def book_table(user_id: str, date: str, time: str, restaurant_location: str, number_of_people: int):
    """Books a table at a restaurant location for a specific date, time, and number of people.

    Args:
        user_id: User ID for the reservation.
        date: Date for the reservation.
        time: Time for the reservation.
        restaurant_location: Restaurant location for the reservation.
        number_of_people: Number of people for the reservation.
    Returns:
        dict: Booking result containing status, message, location, date, time, and guests.
    """
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
                    "✅ Table booked successfully!\n\n"
                    "📍 Location: Connaught Place\n"
                    "📅 Date: 2024-06-14\n"
                    "🕐 Time: 7:00 PM\n"
                    f"👥 Guests: {number_of_people} people"
                ),
                "location": "Connaught Place",
                "date": "2024-06-14",
                "time": "7:00 PM",
                "guests": 4,
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error booking table: {str(e)}",
            "location": restaurant_location,
            "date": date,
            "time": time,
            "guests": number_of_people,
        }


def get_recommendation(user_lat: float, user_long: float):
    """
    Gets restaurant recommendations based on user's location coordinates.
    Args:
        user_lat: User's latitude coordinate.
        user_long: User's longitude coordinate.
    This function finds the nearest restaurant venues to the user's location using
    latitude and longitude coordinates. It calculates distances and returns the top
    recommendations sorted by proximity.
        
    Returns:
        dict: A dictionary containing:
            - status: "success" or "error"
            - recommendations: List of recommended venues with details:
                - name: Venue name
                - address: Full address
                - distance_km: Distance in kilometers
                - coordinates: (latitude, longitude)
            - user_location: User's coordinates
            - count: Number of recommendations returned
    
    Example:
        >>> get_recommendation(28.6315, 77.2167)
        {"status": "success", "recommendations": [...], "user_location": (28.6315, 77.2167), "count": 3}
    """
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
                "message": f"🌟 Top 3 Nearby Venues 🌟\n\n1. Connaught Place\n   📍 Address: Connaught Place, New Delhi\n   📏 Distance: 0.5 km\n\n2. Khan Market\n   📍 Address: Khan Market, New Delhi\n   📏 Distance: 2.1 km\n\n3. Cyber City Gurugram\n   📍 Address: Cyber City, Gurugram\n   📏 Distance: 15.3 km",
                "user_location": (user_lat, user_long),
                "count": 3
            }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting recommendations: {str(e)}",
            "user_location": (user_lat, user_long),
            "count": 0
        }


def get_available_locations():
    """
    Returns a list of all available restaurant locations with their details.
    
    This function provides information about all restaurant locations including
    their names, addresses, coordinates, and capacities. Useful for users
    to see all available options.
    
    Returns:
        dict: A dictionary containing:
            - status: "success"
            - locations: List of location dictionaries with:
                - id: Location ID
                - name: Location name
                - address: Full address
                - coordinates: (latitude, longitude)
                - capacity: Total capacity
            - count: Total number of locations
    
    Example:
        >>> get_available_locations()
        {"status": "success", "locations": [...], "count": 25}
    """
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
            "message": f"Error loading locations: {str(e)}",
            "locations": [],
            "count": 0
        }


AVAILABLE_FUNCTIONS = [
    book_table,
    get_recommendation,
    get_available_locations
]


def check_missing_parameters(function_name: str, provided_args: Dict[str, Any]) -> List[str]:
    """
    Check which required parameters are missing for a given function.
    
    Args:
        function_name (str): Name of the function to check
        provided_args (dict): Dictionary of arguments that were provided
    
    Returns:
        list: List of missing parameter names
    
    Example:
        >>> check_missing_parameters("book_table", {"date": "tomorrow", "time": "7pm"})
        ['user_id', 'restaurant_location', 'number_of_people']
    """
    # Find the function object
    function_obj = None
    for func in AVAILABLE_FUNCTIONS:
        if func.__name__ == function_name:
            function_obj = func
            break
    
    if function_obj is None:
        return []
    
    # Get function signature
    sig = inspect.signature(function_obj)
    required_params = []
    
    # Get all parameters (excluding *args and **kwargs)
    for param_name, param in sig.parameters.items():
        # Check if parameter has a default value
        if param.default == inspect.Parameter.empty:
            required_params.append(param_name)
    
    # Find missing parameters
    missing_params = [param for param in required_params if param not in provided_args or provided_args[param] is None]
    
    return missing_params


def extract_tool_calls(response: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls and arguments from FunctionGemma response.
    
    Handles multiple formats:
    1. JSON format: {"function": "name", "arguments": {...}}
    2. Special tokens: <start_function_call>...<end_function_call>
    3. Function call format: call:function_name{arg1:value1,arg2:value2}
    
    Args:
        response (str): The raw response from FunctionGemma
    
    Returns:
        list: List of dictionaries, each containing:
            - function (str): Function name
            - arguments (dict): Dictionary of arguments
            - raw (str): Raw extracted text
    
    Example:
        >>> extract_tool_calls('<start_function_call>call:book_table{date:"tomorrow",time:"7pm"}<end_function_call>')
        [{'function': 'book_table', 'arguments': {'date': 'tomorrow', 'time': '7pm'}, 'raw': '...'}]
    """
    tool_calls = []
    
    # Method 1: Try to find JSON objects in the response
    json_pattern = r'\{[^{}]*"function"[^{}]*"arguments"[^{}]*\}'
    json_matches = re.finditer(json_pattern, response, re.DOTALL)
    
    for match in json_matches:
        try:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            if "function" in parsed and "arguments" in parsed:
                tool_calls.append({
                    "function": parsed["function"],
                    "arguments": parsed["arguments"],
                    "raw": json_str
                })
        except json.JSONDecodeError:
            continue
    
    # Method 2: Look for FunctionGemma special tokens
    start_token = "<start_function_call>"
    end_token = "<end_function_call>"
    
    if start_token in response and end_token in response:
        pattern = f"{re.escape(start_token)}(.*?){re.escape(end_token)}"
        matches = re.finditer(pattern, response, re.DOTALL)
        
        for match in matches:
            call_text = match.group(1).strip()
            parsed_call = _parse_function_call_text(call_text)
            if parsed_call:
                parsed_call["raw"] = match.group(0)
                tool_calls.append(parsed_call)
    
    # Method 3: Look for call:function_name{...} pattern
    call_pattern = r'call:(\w+)\{([^}]*)\}'
    matches = re.finditer(call_pattern, response)
    
    for match in matches:
        func_name = match.group(1)
        args_text = match.group(2)
        args_dict = _parse_arguments_text(args_text)
        
        tool_calls.append({
            "function": func_name,
            "arguments": args_dict,
            "raw": match.group(0)
        })
    
    # Method 4: Try to find function names followed by JSON-like arguments
    for func in AVAILABLE_FUNCTIONS:
        func_name = func.__name__
        # Look for function name followed by arguments
        pattern = rf'\b{func_name}\s*[\(\[]\s*([^\)\]]+)\s*[\)\]]'
        matches = re.finditer(pattern, response, re.IGNORECASE)
        
        for match in matches:
            args_text = match.group(1)
            args_dict = _parse_arguments_text(args_text)
            tool_calls.append({
                "function": func_name,
                "arguments": args_dict,
                "raw": match.group(0)
            })
    
    return tool_calls


def _parse_function_call_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse function call text in various formats."""
    # Try call:function{args} format
    call_match = re.match(r'call:(\w+)\{([^}]*)\}', text)
    if call_match:
        func_name = call_match.group(1)
        args_text = call_match.group(2)
        args_dict = _parse_arguments_text(args_text)
        return {
            "function": func_name,
            "arguments": args_dict
        }
    
    # Try JSON format
    try:
        parsed = json.loads(text)
        if "function" in parsed:
            return {
                "function": parsed["function"],
                "arguments": parsed.get("arguments", {})
            }
    except json.JSONDecodeError:
        pass
    
    return None


def _parse_arguments_text(args_text: str) -> Dict[str, Any]:
    """Parse arguments from text string."""
    args_dict = {}
    
    if not args_text.strip():
        return args_dict
    
    # Try JSON format first
    try:
        # Wrap in braces to make it valid JSON
        json_str = "{" + args_text + "}"
        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError:
        pass
    
    # Try key:value pairs
    # Handle escaped values with <escape> tags
    args_text = re.sub(r'<escape>(.*?)<escape>', r'"\1"', args_text)
    
    # Try to parse key:value pairs
    # Pattern: key:value or key:"value"
    pattern = r'(\w+):\s*([^,}]+)'
    matches = re.finditer(pattern, args_text)
    
    for match in matches:
        key = match.group(1)
        value = match.group(2).strip()
        
        # Remove quotes if present
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]
        
        # Try to convert to appropriate type
        try:
            # Try int
            if value.isdigit():
                value = int(value)
            # Try float
            elif re.match(r'^-?\d+\.\d+$', value):
                value = float(value)
            # Try boolean
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
        except (ValueError, AttributeError):
            pass
        
        args_dict[key] = value
    
    return args_dict


def analyze_query_parameters(user_query: str, function_name: str) -> Dict[str, Any]:
    """
    Analyze a user query to check which parameters are provided and which are missing.
    
    Args:
        user_query (str): The user's query
        function_name (str): Name of the function to analyze for
    
    Returns:
        dict: Dictionary containing:
            - function (str): Function name
            - provided_params (dict): Parameters found in the query
            - missing_params (list): List of missing parameter names
            - analysis (str): Human-readable analysis
    
    Example:
        >>> analyze_query_parameters("Book a table for 4 people tomorrow", "book_table")
        {
            'function': 'book_table',
            'provided_params': {'number_of_people': '4', 'date': 'tomorrow'},
            'missing_params': ['user_id', 'time', 'restaurant_location'],
            'analysis': 'Found 2 parameters, missing 3 parameters...'
        }
    """
    # First, try to extract tool calls from a hypothetical response
    # This is a simplified analysis - in practice, you'd run the model first
    
    # Find the function
    function_obj = None
    for func in AVAILABLE_FUNCTIONS:
        if func.__name__ == function_name:
            function_obj = func
            break
    
    if function_obj is None:
        return {
            "function": function_name,
            "provided_params": {},
            "missing_params": [],
            "analysis": f"Function '{function_name}' not found"
        }
    
    # Get all parameters from function signature
    sig = inspect.signature(function_obj)
    all_params = list(sig.parameters.keys())
    
    # Simple keyword extraction (this is basic - you might want to use NLP)
    provided_params = {}
    
    # Look for common patterns
    # Number of people
    people_match = re.search(r'(\d+)\s*(?:people|guests|persons|pax)', user_query, re.IGNORECASE)
    if people_match:
        provided_params['number_of_people'] = int(people_match.group(1))
    
    # Date patterns
    date_patterns = [
        r'tomorrow', r'today', r'\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}',
        r'\w+\s+\d{1,2}', r'\d{4}-\d{2}-\d{2}'
    ]
    for pattern in date_patterns:
        if re.search(pattern, user_query, re.IGNORECASE):
            date_match = re.search(pattern, user_query, re.IGNORECASE)
            if date_match:
                provided_params['date'] = date_match.group(0)
                break
    
    # Time patterns
    time_patterns = [
        r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', r'\d{1,2}\s*(?:AM|PM|am|pm)',
        r'\d{1,2}:\d{2}', r'(?:morning|afternoon|evening|night)'
    ]
    for pattern in time_patterns:
        if re.search(pattern, user_query, re.IGNORECASE):
            time_match = re.search(pattern, user_query, re.IGNORECASE)
            if time_match:
                provided_params['time'] = time_match.group(0)
                break
    
    # Location patterns (look for common location names or "at [location]")
    location_match = re.search(r'at\s+([A-Z][a-zA-Z\s]+)', user_query)
    if location_match:
        provided_params['restaurant_location'] = location_match.group(1).strip()
    
    # Check missing parameters
    missing_params = check_missing_parameters(function_name, provided_params)
    
    # Create analysis text
    analysis = f"Function: {function_name}\n"
    analysis += f"Found {len(provided_params)} parameter(s): {list(provided_params.keys())}\n"
    analysis += f"Missing {len(missing_params)} parameter(s): {missing_params}"
    
    return {
        "function": function_name,
        "provided_params": provided_params,
        "missing_params": missing_params,
        "analysis": analysis
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
    """
    Generate a response using FunctionGemma with function calling.
    
    Args:
        user_message (str): User's input message
        tokenizer: FunctionGemma tokenizer
        model: FunctionGemma model
        functions (list): List of function objects to include
        include_location_context (bool): Whether to include location data in context
        max_new_tokens (int): Maximum tokens to generate
        do_sample (bool): Whether to use sampling
        extract_tool_calls_flag (bool): If True, also return extracted tool calls
    
    Returns:
        str or dict: If extract_tool_calls_flag is False, returns the raw response string.
                    If True, returns a dict with 'response' and 'tool_calls' keys.
    """
    if functions is None:
        functions = AVAILABLE_FUNCTIONS
    
    # Prepare messages
    messages = [{"role": "user", "content": user_message}]
    
    # Add location context if requested (for large context testing)
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
    
    # Move to device if model has device attribute
    if hasattr(model, "device"):
        for k in inputs:
            inputs[k] = inputs[k].to(model.device)
    
    # Generate response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    # Decode only the newly generated tokens
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

    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it")
    
    test_message = "I want to book a table for 4 people tomorrow at 7pm"
    
    print(f"User Query: {test_message}")
    print("\n" + "="*80)
    
    # Example 1: Check missing parameters
    print("\n1. Checking missing parameters for 'book_table':")
    analysis = analyze_query_parameters(test_message, "book_table")
    print(f"   {analysis['analysis']}")
    print(f"   Provided: {analysis['provided_params']}")
    print(f"   Missing: {analysis['missing_params']}")
    
    # Example 2: Generate response and extract tool calls
    print("\n2. Generating response with FunctionGemma:")
    result = generate_with_functiongemma(
        user_message=test_message,
        tokenizer=tokenizer,
        model=model,
        include_location_context=False,
        max_new_tokens=256,
        do_sample=False,
        extract_tool_calls_flag=True
    )
    
    print(f"\n   Raw Response: {result['response']}")
    print(f"\n   Extracted Tool Calls: {len(result['tool_calls'])} found")
    
    for i, tool_call in enumerate(result['tool_calls'], 1):
        print(f"\n   Tool Call {i}:")
        print(f"     Function: {tool_call['function']}")
        print(f"     Arguments: {tool_call['arguments']}")
        
        # Check missing parameters for this tool call
        missing = check_missing_parameters(tool_call['function'], tool_call['arguments'])
        if missing:
            print(f"     ⚠️  Missing parameters: {missing}")
        else:
            print(f"     ✅ All required parameters provided")
    
    print("\n" + "="*80)
