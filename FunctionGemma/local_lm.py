import json
import os
import sys
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


def generate_with_functiongemma(
    user_message: str,
    tokenizer,
    model,
    functions=None,
    include_location_context=False,
    max_new_tokens=256,
    do_sample=False
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
    
    Returns:
        str: Generated response from the model
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
    
    return response


if __name__ == "__main__":

    print("Initializing model...")
    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it")
    
    test_message = "I want to book a table for 4 people tomorrow at 7pm at Connaught Place"
    
    print(f"User Query: {test_message}")
    
    response = generate_with_functiongemma(
        user_message=test_message,
        tokenizer=tokenizer,
        model=model,
        include_location_context=False,
        max_new_tokens=256,
        do_sample=False
    )
    
    print("response from functiongemma:" , response)
