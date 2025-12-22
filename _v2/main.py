"""
Streamlit-based chat interface for restaurant table reservations.
Uses OpenRouter LLM with function calling to handle table booking requests.
"""
import os
import sys
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, Any, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.booking_utils import book_table

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is not set in environment variables")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

BOOK_TABLE_TOOL = {
    "type": "function",
    "function": {
        "name": "book_table",
        "description": "Book a table at a restaurant. Use this function when the user wants to make a reservation.",
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "The date for the reservation. Accept formats like 'today', 'tomorrow', 'Dec 25', '25th December', etc."
                },
                "time": {
                    "type": "string",
                    "description": "The time for the reservation. Accept formats like '7pm', '7:00 PM', '19:00', 'evening', etc."
                },
                "restaurant_location": {
                    "type": "string",
                    "description": "The location or name of the restaurant where the user wants to book a table."
                },
                "number_of_people": {
                    "type": "string",
                    "description": "The number of people for the reservation. Should be a number as a string."
                }
            },
            "required": ["date", "time", "restaurant_location", "number_of_people"]
        }
    }
}

SYSTEM_PROMPT = """You are a helpful assistant for a restaurant reservation system. Your job is to help users book tables at restaurants.

When a user wants to book a table, you should:
1. Use the book_table function to make the reservation
2. If any required parameters are missing (date, time, restaurant_location, number_of_people), ask the user for them in a friendly and conversational way
3. Once you have all the required parameters, call the book_table function
4. Be friendly, professional, and helpful

Always ask for missing information before attempting to book a table."""

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

def add_message(role: str, content: str):
    """Add a message to the chat history."""
    st.session_state.messages.append({"role": role, "content": content})
    st.session_state.conversation_history.append({"role": role, "content": content})

def execute_book_table(date: str, time: str, restaurant_location: str, number_of_people: str) -> str:
    """Execute the book_table function and return the result."""
    try:
        user_id = f"streamlit_user_{st.session_state.get('user_id', 'default')}"
        result = book_table(
            user_id=user_id,
            date=date,
            time=time,
            restaurant_location=restaurant_location,
            number_of_people=number_of_people
        )
        return result
    except Exception as e:
        return f"❌ An error occurred while booking the table: {str(e)}"

def handle_function_call(tool_call: Dict[str, Any]) -> str:
    """Handle a function call from the LLM."""
    function_name = tool_call.function.name
    
    if function_name == "book_table":
        # Parse arguments
        try:
            arguments = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return "❌ Error: Invalid function arguments format."
        
        required_params = ["date", "time", "restaurant_location", "number_of_people"]
        missing_params = [param for param in required_params if not arguments.get(param)]
        
        if missing_params:
            return f"I need a few more details to complete your reservation. Could you please provide: {', '.join(missing_params)}?"
        
        result = execute_book_table(
            date=arguments.get("date"),
            time=arguments.get("time"),
            restaurant_location=arguments.get("restaurant_location"),
            number_of_people=arguments.get("number_of_people")
        )
        return result
    else:
        return f"❌ Unknown function: {function_name}"

def get_llm_response(user_message: str) -> str:
    """Get response from OpenRouter LLM with function calling."""
    
    try:
        completion = client.chat.completions.create(
            model="qwen/qwen3-8b", 
            messages=st.session_state.conversation_history,
            tools=[BOOK_TABLE_TOOL],
            tool_choice="auto",  
            temperature=0.7,
            max_tokens=1000
        )
        
        message = completion.choices[0].message
        
        if message.tool_calls:
            function_results = []
            for tool_call in message.tool_calls:
                function_result = handle_function_call(tool_call)
                function_results.append(function_result)
                
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }]
                })
                
                st.session_state.conversation_history.append({
                    "role": "tool",
                    "content": function_result,
                    "tool_call_id": tool_call.id
                })
            
            # Get a follow-up response from the LLM
            follow_up = client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=st.session_state.conversation_history,
                tools=[BOOK_TABLE_TOOL],
                tool_choice="auto",
                temperature=0.7,
                max_tokens=1000
            )
            
            assistant_message = follow_up.choices[0].message.content
            if assistant_message:
                return assistant_message
            else:
                return "\n".join(function_results)
        else:
            assistant_message = message.content
            if assistant_message:
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_message
                })
                return assistant_message
            else:
                return "I apologize, but I didn't receive a valid response. Please try again."
                
    except Exception as e:
        error_msg = f"❌ Error calling OpenRouter API: {str(e)}"
        st.error(error_msg)
        return error_msg

def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Restaurant Table Reservation",
        page_icon="🍽️",
        layout="wide"
    )
    
    st.title("🍽️ Restaurant Table Reservation")
    st.markdown("Chat with our assistant to book a table at your favorite restaurant!")
    
    initialize_session_state()
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if user_input := st.chat_input("Type your message here..."):

        with st.chat_message("user"):
            st.markdown(user_input)
        
        add_message("user", user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_llm_response(user_input)
                st.markdown(response)
        
        add_message("assistant", response)
    
    # Sidebar with reset button
    with st.sidebar:
        st.header("Options")
        if st.button("🔄 Reset Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_history = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ]
            st.rerun()
        
        st.markdown("---")
        st.markdown("### How to use:")
        st.markdown("""
        1. Start by saying you want to book a table
        2. Provide details like:
           - Date (e.g., "tomorrow", "Dec 25")
           - Time (e.g., "7pm", "7:00 PM")
           - Location (restaurant name or location)
           - Number of people
        3. The assistant will ask for any missing information
        4. Once all details are provided, your table will be booked!
        """)

if __name__ == "__main__":
    main()

