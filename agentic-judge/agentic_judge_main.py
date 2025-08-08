import json
import re
from langchain_core.messages import HumanMessage

def format_training_batches(df, batch_size=10):
    """
    Splits the dataframe into batches and formats each batch for evaluation prompts.

    Args:
        df (pd.DataFrame): Must contain 'English', 'Filipino-Flawed', and 'Filipino-Correct' columns.
        batch_size (int): Number of rows per batch.

    Returns:
        List[Tuple[str, List[Dict]]]: Each tuple contains a formatted prompt and the corresponding rows.
    """
    rows = df.to_dict(orient='records')
    batches = []

    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        prompt_lines = [
            "You are evaluating English-to-Filipino translations based on three criteria:",
            "1. Adequacy: Does the Filipino translation preserve the meaning of the original sentence?",
            "2. Fluency: Is it natural, smooth, and grammatically correct to be easily understood by a native speaker?",
            "3. Lexical Choice: Are the words contextually accurate and culturally approrpiate?",
            "",
            "Please provide scores from 1-5, with 1 being the lowest and 5 being the highest, along with reasoning for each.",
            "Respond ONLY with a valid JSON array. Do not use markdown code blocks (```). Do not include any explanations, formatting, or text outside of the JSON. Start your response directly with [ and end with ]."
            "Your response must look exactly like this:",
            "[",
                "{",
                    "'item_num': 0,",
                    "'adequacy_score': 4,"
                    "'adequacy_reasoning': 'The correct translation preserves the meaning well.'",
                    "'fluency_score': 5",
                    "'fluency_reasoning': 'The sentence reads smoothly in Filipino.'",
                    "'lexical_choice_score': 4",
                    "'lexical_choice_reasoning': 'Some terms could be slightly more precise.'",
                "}",
               " ...",
            "]",
            "",
            "Items:\n",
        ]

        for idx, row in enumerate(batch, start=1):
            prompt_lines.append(f"{idx}.\nEnglish: {row['English']}")
            prompt_lines.append(f"Flawed Translation: {row['Filipino-Flawed']}")
            prompt_lines.append(f"Correct Translation: {row['Filipino-Correct']}")
            prompt_lines.append("")

        prompt_text = "\n".join(prompt_lines)
        batches.append((prompt_text, batch))

    return batches


def evaluate_training_batches(batches, llm):
    """
    Sends each formatted prompt batch to the LLM and parses the JSON response.

    Args:
        batches (List[Tuple[str, List[Dict]]]): Output from format_translation_batches.
        llm (object): LLM object with an `.invoke()` method (e.g., Gemini, OpenAI wrapper).

    Returns:
        List[Dict]: Evaluation results per item.
    """
    all_evaluations = []

    for prompt_text, batch_rows in batches:
        try:
            # FIX: Wrap the prompt text in a HumanMessage object
            message = HumanMessage(content=prompt_text)
            response = llm.invoke([message])  # Pass as a list of messages
            
            # Extract the content from the response
            response_content = response.content
            
            # Parse the JSON response
            parsed = json.loads(response_content)

            for result in parsed:
                item_index = result.get("item_num", 1) - 1 
                if item_index >= len(batch_rows):
                    continue 
                    
                original_row = batch_rows[item_index]

                evaluation = {
                    "orig_sentence": original_row["English"],
                    "flawed_translation": original_row["Filipino-Flawed"],
                    "target_translation": original_row["Filipino-Correct"],
                    **result
                }

                all_evaluations.append(evaluation)

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {response.content[:500]}...")
            all_evaluations.append({
                "error": f"JSON parsing error: {str(e)}",
                "prompt": prompt_text[:100] + "..." 
            })
            
        except Exception as e:
            print(f"General error: {e}")
            all_evaluations.append({
                "error": str(e),
                "prompt": prompt_text[:100] + "..." 
            })

    return all_evaluations


def clean_json_response(response_content):
    """
    Clean the response content to extract valid JSON from markdown code blocks
    """
    # Remove markdown code blocks if present
    if response_content.strip().startswith('```'):
        # Extract content between ```json and ``` or between ``` and ```
        patterns = [
            r'```json\s*\n(.*?)\n```',  # ```json ... ```
            r'```\s*\n(.*?)\n```',      # ``` ... ```
            r'```json(.*?)```',         # ```json...``` (no newlines)
            r'```(.*?)```'              # ```...``` (no newlines)
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_content, re.DOTALL)
            if match:
                return match.group(1).strip()
    
    # If no code blocks, return as is
    return response_content.strip()