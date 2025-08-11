from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_chroma import Chroma
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm
from typing import Dict, Any

import pandas as pd
import requests
import json
import re

class TranslationEvaluation(BaseModel):
    """Structued output schema for translation evaluations"""

    overall_score: int = Field(description="Overall translation quality score (1-5)", ge=1, le=5)
    adequacy_score: int = Field(description="How well meaning is preserved (1-5)", ge=1, le=5)
    fluency_score: int = Field(description="Natural and grammatical correctness (1-5)", ge=1, le=5)
    lexical_choice_score: int = Field(description="Word choice appropriateness (1-5)", ge=1, le=5)
    adequacy_explanation: str = Field(description="Detailed reasoning for adequacy score")
    fluency_explanation: str = Field(description="Detailed reasoning for fluency score")
    lexical_choice_explanation: str = Field(description="Detailed reasoning for lexical choice score")
    overall_explanation: str = Field(description="Summary explanation for overall score")
    similar_examples_influence: str = Field(description="How similar examples influenced the evaluation")

    @field_validator('overall_score', 'adequacy_score', 'fluency_score', 'lexical_choice_score')
    def validate_scores(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v
    

@tool(description="Translate English sentences to Filipino using LibreTranslate")
def libretranslate_en_fil(text: str) -> str:
    """Translates a sentence in English to Tagalog."""
    res = requests.post("http://127.0.0.1:5000/translate", json={
        "q": text, "source": "en", "target": "tl"
    })
    res.raise_for_status()
    return res.json()["translatedText"]

def format_training_batches(df, batch_size):
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
            "3. Lexical Choice: Are the words contextually accurate and culturally appropriate?",
            "",
            "Please provide scores from 1-5, with 1 being the lowest and 5 being the highest, along with detailed reasoning for each that cites words or phrases that may make the translation flawed.",
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



    for batch_idx, (prompt_text, batch_rows) in tqdm(enumerate(batches),
                                                     total=len(batches),
                                                     desc='Processing Batches',
                                                     unit='batch'):
        try:
            message = HumanMessage(content=prompt_text)
            response = llm.invoke([message])  
            response_content = response.content
            
            print(f"Response Metadata: {response.usage_metadata}")

            # print(f"RAW JSON Content: {response_content}")

            # Clean and Parse the JSON response
            cleaned_json = clean_json_response(response_content)
            parsed = json.loads(cleaned_json)

            # print(f"Cleaned JSON content: {json}")

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

            tqdm.write(f"Batch {batch_idx + 1}: {len(parsed)} items processed")   

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


def get_similar_examples(english_input: str, vector_store: Chroma, k: int = 3):
    # Embed the English input
    similar_examples = []
    nearest_docs = vector_store.similarity_search(english_input, k=k)
    for doc in nearest_docs:
        content = doc.page_content
        metadata = doc.metadata
        similar_examples.append(
            f"""English Sentence: {content}\n
                Flawed Filipino Translation: {metadata['flawed_translation']}\n
                Correct Filipino Translation: {metadata['correct_translation']}\n
                Adequacy [{metadata['adequacy_score']}] - {metadata['adequacy_reasoning']}\n
                Fluency [{metadata['fluency_score']}] - {metadata['fluency_reasoning']}\n
                Lexical Choice [{metadata['lexical_score']}] - {metadata['lexical_reasoning']}
            """
        )

    return similar_examples


# Without dynamic semantic similarity search embedded within the prompt
def prepare_few_shot_agent_with_tools(nearest_examples, llm, tools):
    system_message = f"""
    You are a professional translation evaluator. You must assess a Filipino translation based on:
    - Adequacy: Does the Filipino translation preserve the meaning of the original sentence?.
    - Fluency: Is it natural, smooth, and grammatically correct to be easily understood by a native speaker?.
    - Lexical Choice: Are the words contextually accurate and culturally appropriate?.

    Here are examples of translations and ratings to guide your evaluation:
    {nearest_examples[0]}
    {nearest_examples[1]}
    {nearest_examples[2]}

    For each input:
    - First, explain how similar sentences and translations influence your judgment.
    - Then translate the English sentence using LibreTranslate to serve as partial basis for your rating.  
    - Then give:
      - Adequacy rating (1-5) + detailed reasoning for your score (cite words or phrases from the translation),
      - Fluency rating (1-5) + reasoning ,
      - Lexical Choice rating (1-5) + reasoning.
      - Overall rating (1-5),

    All the reasonings should be detailed and should especially cite words or phrases from the translation that make the translation flawed, if applicable.
    Output Format:
    English Sentence: ...
    Filipino Translation: ...

    Similar Examples Influence: [How did the examples guide your decision?]
    Adequacy: (1-5) + [reasoning citing specific words/phrases]
    Fluency: (1-5) + [reasoning citing specific words/phrases]
    Lexical Choice: (1-5) + [reasoning citing specific words/phrases]
    Overall Rating: ...

    Do not preamble.
    """
    
    system_prompt = SystemMessagePromptTemplate.from_template(system_message)

    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)

    return agent_executor


def prepare_structured_agent_with_tools(system_message: str, llm, tools, parser):
    """Prepare agent executor with structured output capability"""
    
    system_prompt = SystemMessagePromptTemplate.from_template(system_message)
    
    prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        return_intermediate_steps=True,
        handle_parsing_errors=True
    )
    
    return agent_executor


def evaluate_with_agent_executor(llm, english, filipino, agent_executor, vector_store, tools):
    nearest_examples = get_similar_examples(english, vector_store)
    agent_executor = prepare_few_shot_agent_with_tools(nearest_examples, llm, tools)
    response = agent_executor.invoke({"input": f"English Sentence: {english}\nFilipino Translation: {filipino}"}, return_only_outputs=True)
    return response['output']


def structured_evaluate_with_agent_executor(english: str, filipino: str, llm, vector_store, tools) -> Dict[str, Any]:
    """
    Takes english-filipino translation pair and outputs structured evaluation in JSON form.

    Args:
        english (str): English sentence to be translated.
        filipino (str): Filipino translation to be evaluated.
        llm: Language model instance
        vector_store: Vector store for similarity search
        tools: List of tools including LibreTranslate

    Returns:
        Dict[str, Any]: Structured evaluation results in JSON format
    """
    
    # Set up the output parser
    parser = PydanticOutputParser(pydantic_object=TranslationEvaluation)

    # Get similar examples from vector store
    nearest_examples = get_similar_examples(english, vector_store, k=3)

    # Create the structured prompt
    system_message = f"""
    You are a professional translation evaluator. You must assess a Filipino translation based on:
    - Adequacy: Does the Filipino translation preserve the meaning of the original sentence?
    - Fluency: Is it natural, smooth, and grammatically correct to be easily understood by a native speaker?
    - Lexical Choice: Are the words contextually accurate and culturally appropriate?

    Here are examples of translations and ratings to guide your evaluation:
    {nearest_examples[0] if nearest_examples else "No similar examples found"}
    {nearest_examples[1] if len(nearest_examples) > 1 else ""}
    {nearest_examples[2] if len(nearest_examples) > 2 else ""}

    EVALUATION PROCESS:
    1. First, analyze how the similar examples influence your judgment.
    2. Use LibreTranslate to get an alternative translation for comparison.
    3. Evaluate the given translation on all three criteria.
    4. Provide detailed reasoning for your thought process, citing specific words/phrases.

    {parser.get_format_instructions()}
    
    Be extremely detailed in your explanations and cite specific words or phrases from the translation.
    """

    # Create the agent with structured output
    agent_executor = prepare_few_shot_agent_with_tools(system_message, llm, tools)

    try:
        # Execute the evaluation
        input_text = f"English Sentence: {english}\nFilipino Translation: {filipino}"
        response = agent_executor.invoke({"input": input_text})
        
        # Parse the structured output
        if isinstance(response.get('output'), str):
            # Try to parse JSON from string output
            cleaned_output = clean_json_response(response['output'])
            try:
                structured_result = json.loads(cleaned_output)
            except json.JSONDecodeError:
                # Fallback to regex parsing if JSON parsing fails
                structured_result = fallback_parse_evaluation(response['output'])
        else:
            structured_result = response.get('output', {})
        
        # Ensure all required fields are present
        structured_result = validate_and_complete_output(structured_result, english, filipino)
        
        # Add metadata
        structured_result['metadata'] = {
            'english_sentence': english,
            'filipino_translation': filipino,
            'evaluation_timestamp': str(pd.Timestamp.now()),
            'model_used': str(type(llm).__name__),
            'similar_examples_count': len(nearest_examples)
        }
        
        return structured_result
        
    except Exception as e:
        # Return error structure if evaluation fails
        return create_error_output(english, filipino, str(e))


def fallback_parse_evaluation(response_text: str) -> Dict[str, Any]:
    """Fallback parser using regex if structured parsing fails"""
    def extract_score(pattern: str, text: str, default: int = 3) -> int:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                return default
        return default
    
    def extract_explanation(pattern: str, text: str, default: str = "No explanation provided") -> str:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            explanation = match.group(1).strip()
            # Remove score number at beginning (e.g., "3 - The translation...")
            explanation = re.sub(r'^\d+\s*-\s*', '', explanation)
            return explanation
        return default
    
    return {
        "overall_score": extract_score(r"overall.*?rating.*?(\d)", response_text),
        "adequacy_score": extract_score(r"adequacy.*?(\d)", response_text),
        "fluency_score": extract_score(r"fluency.*?(\d)", response_text),
        "lexical_choice_score": extract_score(r"lexical.*?choice.*?(\d)", response_text),
        
        "adequacy_explanation": extract_explanation(
            r"adequacy\s*:\s*\d+\s*-\s*(.*?)(?=\nfluency|\nlexical|\noverall|$)", 
            response_text
        ),
        "fluency_explanation": extract_explanation(
            r"fluency\s*:\s*\d+\s*-\s*(.*?)(?=\nlexical|\noverall|$)", 
            response_text
        ),
        "lexical_choice_explanation": extract_explanation(
            r"lexical\s+choice\s*:\s*\d+\s*-\s*(.*?)(?=\noverall|$)", 
            response_text
        ),
        "similar_examples_influence": extract_explanation(
            r"similar.*?examples.*?influence\s*:(.*?)(?=\nadequacy|\nfluency|\nlexical|$)", 
            response_text
        ),
    }


def validate_and_complete_output(result: Dict[str, Any], english: str, filipino: str) -> Dict[str, Any]:
    """Ensure all required fields are present and valid"""
    # Required fields with defaults
    required_fields = {
        "overall_score": 0,
        "adequacy_score": 0,
        "fluency_score": 0,
        "lexical_choice_score": 0,
        "adequacy_explanation": "No explanation provided",
        "fluency_explanation": "No explanation provided", 
        "lexical_choice_explanation": "No explanation provided",
        "similar_examples_influence": "No similar examples analysis",
    }
    
    # Fill missing fields
    for field, default in required_fields.items():
        if field not in result or result[field] is None:
            result[field] = default
    
    # Validate scores are in range 1-5
    score_fields = ["overall_score", "adequacy_score", "fluency_score", "lexical_choice_score"]
    for field in score_fields:
        try:
            score = int(result[field])
            result[field] = max(1, min(5, score))  # Clamp to 1-5 range
        except (ValueError, TypeError):
            result[field] = 3  # Default to middle score
    
    return result


def create_error_output(english: str, filipino: str, error_message: str) -> Dict[str, Any]:
    """Create structured error output when evaluation fails"""
    return {
        "overall_score": 1,
        "adequacy_score": 1,
        "fluency_score": 1,
        "lexical_choice_score": 1,
        "adequacy_explanation": f"Evaluation failed: {error_message}",
        "fluency_explanation": f"Evaluation failed: {error_message}",
        "lexical_choice_explanation": f"Evaluation failed: {error_message}",
        "overall_explanation": f"Evaluation failed: {error_message}",
        "similar_examples_influence": "Could not retrieve similar examples due to error",
        "metadata": {
            "english_sentence": english,
            "filipino_translation": filipino,
            "evaluation_timestamp": str(pd.Timestamp.now()),
            "error": True,
            "error_message": error_message
        }
    }


def clean_json_response(response_content):
    """
    Cleans LLM output by removing markdown code blocks and fixes common JSON issues.
    
    Args:
        response_content (str): Raw string output from the LLM.
        
    Returns:
        str: Cleaned JSON string ready for json.loads().
    """
    import re

    # Strip leading/trailing whitespace
    text = response_content.strip()

    if response_content.strip().endswith("..."):
        print("Truncated response detected. Reduce batch size or prompt verbosity.")

    # Remove markdown code block fences like ```json ... ```
    if text.startswith("```json"):
        text = re.sub(r"^```json", "", text)
    if text.startswith("```"):
        text = re.sub(r"^```", "", text)
    if text.endswith("```"):
        text = re.sub(r"```$", "", text)

    # Remove trailing commas, which are illegal in JSON
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()


def print_collection_records(collection, limit=5):
    records = collection.get(
        include=["embeddings", "documents", "metadatas"],
        limit=limit
    )   

    for doc_id, doc_txt, meta, embed in zip(
        records["ids"],
        records["documents"],
        records["metadatas"],
        records["embeddings"],
    ):
        print("ID:", doc_id)
        print("Doc:", doc_txt[:60], "…")
        print("Meta:", meta)
        print("Embedding size:", len(embed))
        print("First 8 floats:", embed[:8])
        print("—" * 40)


def parse_rating(text):
    m = re.search(r"Rating:\s*([1-5])", text)
    return int(m.group(1)) if m else None


def parse_overall_rating(text):
    m = re.search(r"Overall Rating:\s*([1-5])", text)
    return int(m.group(1)) if m else None