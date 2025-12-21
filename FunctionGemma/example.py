from local_lm import (
    initialize_model,
    generate_with_functiongemma,
    AVAILABLE_FUNCTIONS
)

def main():
    
    print("\nLoading FunctionGemma model...")
    tokenizer, model = initialize_model()
    
    user_query = "I want to book a table for 4 people tomorrow at 7pm at Connaught Place"
    
    print(f"User Query: {user_query}")
    
    # Generate response
    response = generate_with_functiongemma(
        user_query,
        tokenizer,
        model,
        functions=AVAILABLE_FUNCTIONS,
        include_location_context=False,
        max_new_tokens=256
    )
    
    print(response)

if __name__ == "__main__":
    main()