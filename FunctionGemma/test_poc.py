"""
FunctionGemma POC Test Script

This script tests FunctionGemma's capabilities with various scenarios:
1. Simple function calls
2. Complex multi-parameter function calls
3. Large context handling (with location data)
4. Argument extraction and management
5. Edge cases and ambiguous queries
"""

import sys
import os
from local_lm import (
    initialize_model,
    generate_with_functiongemma,
    AVAILABLE_FUNCTIONS,
    get_location_context
)

# Test scenarios
TEST_SCENARIOS = [
    {
        "name": "Simple Booking Request",
        "message": "Book a table for 4 people tomorrow at 7pm at Connaught Place",
        "include_context": False,
        "description": "Tests basic function calling with all parameters provided"
    },
    {
        "name": "Partial Parameters",
        "message": "I want to book a table for 2 people",
        "include_context": False,
        "description": "Tests function calling with missing parameters"
    },
    {
        "name": "Recommendation Request",
        "message": "I'm at 28.6315, 77.2167. Can you recommend nearby restaurants?",
        "include_context": False,
        "description": "Tests location-based recommendation function"
    },
    {
        "name": "Large Context - Booking with Location Data",
        "message": "Book a table for 6 people on December 25th at 8pm at Khan Market",
        "include_context": True,
        "description": "Tests function calling with large context (all location data included)"
    },
    {
        "name": "Ambiguous Query",
        "message": "I need a reservation for dinner",
        "include_context": False,
        "description": "Tests handling of ambiguous queries with missing information"
    },
    {
        "name": "Complex Query with Multiple Intents",
        "message": "What restaurants are near me at 28.5445, 77.1926? Also, can I book a table for tomorrow?",
        "include_context": False,
        "description": "Tests handling multiple intents in one query"
    },
    {
        "name": "Natural Language Time",
        "message": "I'd like to reserve a table for 4 guests this evening around 7:30 PM at Cyber City Gurugram",
        "include_context": False,
        "description": "Tests natural language time parsing"
    },
    {
        "name": "Date Variations",
        "message": "Book a table for 3 people on 25th December at 6pm at Hauz Khas",
        "include_context": False,
        "description": "Tests various date format handling"
    },
    {
        "name": "Large Context - Recommendation with All Locations",
        "message": "I'm currently at Connaught Place area (28.6315, 77.2167). Show me the nearest restaurants.",
        "include_context": True,
        "description": "Tests recommendation with full location context"
    },
    {
        "name": "Edge Case - Very Large Party",
        "message": "I need to book a table for 20 people tomorrow at 7pm. Which location can accommodate us?",
        "include_context": True,
        "description": "Tests capacity-aware booking with large context"
    }
]


def print_separator():
    """Print a visual separator"""
    print("\n" + "="*80 + "\n")


def run_test_scenario(scenario, tokenizer, model, index):
    """Run a single test scenario"""
    print_separator()
    print(f"TEST {index + 1}: {scenario['name']}")
    print(f"Description: {scenario['description']}")
    print(f"Include Location Context: {scenario['include_context']}")
    print(f"\nUser Message: {scenario['message']}")
    print("\n" + "-"*80)
    print("FunctionGemma Response:")
    print("-"*80)
    
    try:
        response = generate_with_functiongemma(
            scenario['message'],
            tokenizer,
            model,
            include_location_context=scenario['include_context'],
            max_new_tokens=512,  # Increased for complex responses
            do_sample=False
        )
        print(response)
        print("\n✓ Test completed successfully")
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def analyze_context_size():
    """Analyze the size of location context"""
    print_separator()
    print("CONTEXT SIZE ANALYSIS")
    print("-"*80)
    
    location_context = get_location_context()
    context_length = len(location_context)
    context_lines = len(location_context.split('\n'))
    
    print(f"Location Context Length: {context_length} characters")
    print(f"Location Context Lines: {context_lines} lines")
    print(f"Estimated Tokens (rough): ~{context_length // 4} tokens")
    print("\nThis context will be included in tests marked with 'include_context: True'")


def main():
    """Main test runner"""
    print("="*80)
    print("FunctionGemma POC - Restaurant Reservation System")
    print("Testing Function Calling with Large Context and Argument Management")
    print("="*80)
    
    # Initialize model
    print("\nInitializing FunctionGemma model...")
    tokenizer, model = initialize_model()
    
    if tokenizer is None or model is None:
        print("ERROR: Failed to load FunctionGemma model.")
        print("Please ensure:")
        print("1. transformers library is installed: pip install transformers")
        print("2. You have internet connection to download the model")
        print("3. Sufficient disk space for the model (~1GB)")
        sys.exit(1)
    
    # Show available functions
    print_separator()
    print("AVAILABLE FUNCTIONS:")
    print("-"*80)
    for i, func in enumerate(AVAILABLE_FUNCTIONS, 1):
        print(f"{i}. {func.__name__}")
        print(f"   {func.__doc__.split(chr(10))[0] if func.__doc__ else 'No description'}")
    
    # Analyze context size
    analyze_context_size()
    
    # Run all test scenarios
    print_separator()
    print(f"Running {len(TEST_SCENARIOS)} test scenarios...")
    print("="*80)
    
    for i, scenario in enumerate(TEST_SCENARIOS):
        run_test_scenario(scenario, tokenizer, model, i)
    
    # Summary
    print_separator()
    print("TEST SUMMARY")
    print("-"*80)
    print(f"Total Tests: {len(TEST_SCENARIOS)}")
    print(f"Tests with Large Context: {sum(1 for s in TEST_SCENARIOS if s['include_context'])}")
    print(f"Tests with Simple Context: {sum(1 for s in TEST_SCENARIOS if not s['include_context'])}")
    print("\n✓ All tests completed!")
    print("\nKey Observations:")
    print("- Check how FunctionGemma handles function argument extraction")
    print("- Observe behavior with large context (location data)")
    print("- Note how it manages missing or ambiguous parameters")
    print("- Evaluate response quality and function selection accuracy")


if __name__ == "__main__":
    main()

