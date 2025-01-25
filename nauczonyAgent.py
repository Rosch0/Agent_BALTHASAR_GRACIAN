from transformers import AutoTokenizer, AutoModelForCausalLM

class FalconModelTester:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir)

    def generate_response(self, user_input):
        # Tokenize the input
        inputs = self.tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate response
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=100,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,  # For more creative responses
            top_k=50,        # Limit the search space
            top_p=0.95       # Nucleus sampling
        )
        
        # Decode the generated response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def interactive_test(self):
        print("Model is ready for testing! Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            response = self.generate_response(user_input)
            print(f"Bot: {response}")

if __name__ == "__main__":
    model_dir = r"C:\Users\Blaze\Desktop\AgentAI\checkpoint-165"

    tester = FalconModelTester(model_dir)
    tester.interactive_test()
