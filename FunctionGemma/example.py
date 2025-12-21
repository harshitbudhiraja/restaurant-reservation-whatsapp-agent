from local_lm import (
    generate_with_functiongemma,
    AVAILABLE_FUNCTIONS
)
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    
    print("\nLoading FunctionGemma model...")
    tokenizer = AutoTokenizer.from_pretrained("google/functiongemma-270m-it")
    model = AutoModelForCausalLM.from_pretrained("google/functiongemma-270m-it")
    
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