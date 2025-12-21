# FunctionGemma POC - Restaurant Reservation System

This directory contains a Proof of Concept (POC) implementation using Google's **FunctionGemma** model for function calling in the restaurant reservation system.

## Overview

This POC demonstrates how FunctionGemma handles:
1. **Function Calling**: Automatic function selection and argument extraction from natural language
2. **Large Context**: Testing with extensive location data (25+ restaurant locations)
3. **Argument Management**: How the model handles complex, multi-parameter functions
4. **Edge Cases**: Ambiguous queries, missing parameters, and natural language variations

## Files

- **`local_lm.py`**: Main implementation with FunctionGemma integration and all restaurant functions
- **`example.py`**: Simple example demonstrating basic usage
- **`test_poc.py`**: Comprehensive test suite with 10+ scenarios
- **`README.md`**: This file

## Functions Available

The POC includes the following functions from the main codebase:

1. **`book_table(user_id, date, time, restaurant_location, number_of_people)`**
   - Books a restaurant table reservation
   - Handles flexible date/time formats
   - Validates capacity and location

2. **`get_recommendation(user_lat, user_long)`**
   - Gets nearby restaurant recommendations based on coordinates
   - Returns top 3 closest venues with distances

3. **`get_available_locations()`**
   - Lists all available restaurant locations
   - Returns location details (name, address, coordinates, capacity)

4. **`get_today_date()`**
   - Helper function to get current date
   - Returns formatted date strings

## Setup

### Prerequisites

```bash
pip install transformers torch
```

### Model

FunctionGemma will automatically download the model on first run:
- Model: `google/functiongemma-270m-it`
- Size: ~1GB
- Requires internet connection for first download

## Usage

### Simple Example

```bash
python example.py
```

This runs a basic example with a single booking query.

### Comprehensive Testing

```bash
python test_poc.py
```

This runs 10+ test scenarios including:
- Simple function calls
- Complex multi-parameter calls
- Large context tests (with location data)
- Ambiguous queries
- Edge cases

### Programmatic Usage

```python
from local_lm import initialize_model, generate_with_functiongemma, AVAILABLE_FUNCTIONS

# Initialize model
tokenizer, model = initialize_model()

# Generate response
response = generate_with_functiongemma(
    "Book a table for 4 people tomorrow at 7pm at Connaught Place",
    tokenizer,
    model,
    functions=AVAILABLE_FUNCTIONS,
    include_location_context=False,
    max_new_tokens=256
)

print(response)
```

## Test Scenarios

The `test_poc.py` includes the following test scenarios:

1. **Simple Booking Request** - All parameters provided
2. **Partial Parameters** - Missing some parameters
3. **Recommendation Request** - Location-based recommendations
4. **Large Context - Booking** - Booking with full location data
5. **Ambiguous Query** - Unclear user intent
6. **Complex Query** - Multiple intents in one query
7. **Natural Language Time** - Flexible time parsing
8. **Date Variations** - Different date formats
9. **Large Context - Recommendation** - Recommendations with full context
10. **Edge Case - Large Party** - Capacity-aware booking

## Large Context Testing

To test how FunctionGemma handles large contexts, the POC includes all 25 restaurant locations in the context. This adds:
- ~2000+ characters of location data
- ~500+ estimated tokens
- Tests model's ability to extract relevant information from large contexts

Enable large context by setting `include_location_context=True` in `generate_with_functiongemma()`.

## Key Observations

When running the POC, observe:

1. **Function Selection**: Does FunctionGemma correctly identify which function to call?
2. **Argument Extraction**: How accurately does it extract parameters from natural language?
3. **Large Context Handling**: Does it maintain accuracy with extensive location data?
4. **Missing Parameters**: How does it handle incomplete information?
5. **Natural Language**: Can it parse various date/time/location formats?

## Integration with Main Codebase

The POC can optionally use actual functions from the main codebase:
- If imports succeed, uses real `book_table()` and `get_recommendation()` functions
- If imports fail, uses mock implementations for testing
- This allows testing without full system dependencies (Redis, WhatsApp, etc.)

## Notes

- FunctionGemma is a 270M parameter model optimized for function calling
- It uses a special chat template that includes function definitions
- The model generates function calls in a structured format
- Response quality depends on prompt clarity and function documentation

## Troubleshooting

**Model download fails:**
- Check internet connection
- Ensure sufficient disk space (~1GB)
- Try: `huggingface-cli login` if authentication is required

**Import errors:**
- The POC works with mock functions if main codebase imports fail
- For full integration, ensure main codebase dependencies are installed

**Out of memory:**
- FunctionGemma-270m is relatively small, but ensure you have ~2GB RAM available
- Consider using CPU if GPU memory is limited

## References

- [FunctionGemma Documentation](https://ai.google.dev/gemma/docs/functiongemma)
- [HuggingFace Model Card](https://huggingface.co/google/functiongemma-270m-it)

