class AdvancedPromptTemplate:
    def __init__(self, task_description="", few_shot_examples=None, scratch_space=None):
        """
        Initialize the prompt template.
        
        :param task_description: A string describing the task.
        :param few_shot_examples: A list of dictionaries, where each dictionary represents an example with 'input', 'reasoning', and 'output' fields.
        :param scratch_space: A flag to indicate whether to include a scratch space or not.
        """
        self.task_description = task_description
        self.few_shot_examples = few_shot_examples if few_shot_examples else []
        self.scratch_space = scratch_space
        
    def add_example(self, input_text, reasoning=None, output_text=None):
        """
        Add an example to the few-shot examples.
        
        :param input_text: The input or question.
        :param reasoning: Step-by-step reasoning (optional for chain-of-thought prompting).
        :param output_text: The final output or answer.
        """
        example = {
            'input': input_text,
            'reasoning': reasoning,
            'output': output_text
        }
        self.few_shot_examples.append(example)
        
    def build_prompt(self, user_input):
        """
        Build the final prompt input for the LLM.
        
        :param user_input: The user's input or query.
        :return: The structured prompt as a string.
        """
        prompt = ""

        # Add task description if provided
        if self.task_description:
            prompt += f"Task: {self.task_description}\n\n"
        
        # Add few-shot examples if provided
        if self.few_shot_examples:
            prompt += "Here are some examples:\n"
            for example in self.few_shot_examples:
                prompt += f"Q: {example['input']}\n"
                if example['reasoning']:
                    prompt += f"Scratch space:\n{example['reasoning']}\n"
                if example['output']:
                    prompt += f"A: {example['output']}\n"
                prompt += "\n"  # Add space between examples
        
        # Add the user query
        prompt += f"Q: {user_input}\n"
        
        # Include scratch space if enabled
        if self.scratch_space:
            prompt += "Scratch space:\n"
        
        # Add the final output placeholder
        prompt += "A:"
        
        return prompt



# Create an instance of the prompt template for solving math problems
math_prompt = AdvancedPromptTemplate(
    task_description="Solve math problems step-by-step.",
    scratch_space=True  # Enable scratch space for reasoning
)

# Add few-shot examples with reasoning
math_prompt.add_example(
    input_text="If you have 5 apples and give away 2, how many do you have left?",
    reasoning="Step 1: Start with 5 apples. Step 2: Subtract 2 from 5. Step 3: 5 - 2 = 3.",
    output_text="3"
)

math_prompt.add_example(
    input_text="A car travels 60 miles per hour. How far does it travel in 2 hours?",
    reasoning="Step 1: The car travels 60 miles in 1 hour. Step 2: Multiply 60 by 2 to get the distance. Step 3: 60 * 2 = 120.",
    output_text="120 miles"
)

# Define the user's query
user_input = "A train travels 50 miles per hour. How far does it travel in 3 hours?"

# Build the final prompt
final_prompt = math_prompt.build_prompt(user_input)
print(final_prompt)

