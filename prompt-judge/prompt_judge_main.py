from langchain_core.prompts import ChatPromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any

import pandas as pd
import json
import re

# Define the structured output schema (simplified version)
class TranslationEvaluation(BaseModel):
    """Structured output schema for basic translation evaluation"""
    overall_score: int = Field(description="Overall translation quality score (1-5)", ge=1, le=5)
    adequacy_score: int = Field(description="How well meaning is preserved (1-5)", ge=1, le=5)
    fluency_score: int = Field(description="Natural and grammatical correctness (1-5)", ge=1, le=5)
    lexical_choice_score: int = Field(description="Word choice appropriateness (1-5)", ge=1, le=5)
    adequacy_explanation: str = Field(description="Detailed reasoning for adequacy score")
    fluency_explanation: str = Field(description="Detailed reasoning for fluency score")
    lexical_choice_explanation: str = Field(description="Detailed reasoning for lexical choice score")
    overall_explanation: str = Field(description="Summary explanation for overall score")

    @field_validator('overall_score', 'adequacy_score', 'fluency_score', 'lexical_choice_score')
    def validate_scores(cls, v):
        if not 1 <= v <= 5:
            raise ValueError('Score must be between 1 and 5')
        return v

# Standardized method to load training set (complete with data preprocessing)
def load_training_set(path):
    """Standardized method for loading training set, complete with data preprocessing.

    Args:
        path (str): string containing path to training set .csv file.

    Returns:
        pd.DataFrame: Dataframe containing training set.
    """
    training_df = pd.read_csv(path).dropna(how='all')
    training_df.drop(columns=['Contributor'], inplace=True)
    training_df['Remarks'] = training_df['Remarks'].fillna("None")

    return training_df

# Standardized method to load test set (complete with data preprocessing)
def load_test_set(path):
    test_df = pd.read_csv(path).dropna(how='all')
    test_df.drop(columns=["Contributor"], inplace=True)

    test_df.rename(columns={
        "Source Text (English)": "English",
        "Target Text (Filipino)": "Filipino",
        "Final Score                          (1 - lowest, 5 - highest)": "Rating",
        "Rater 1 Explanation": "Remarks 1",
        "Rater 2 Explanation": "Remarks 2"
    }, inplace=True)

    test_df['Remarks 2'] = test_df['Remarks 2'].fillna("None")

    return test_df


def prepare_zeroshot_prompt():
    """Original zero-shot prompt (for backward compatibility)"""
    system_message = """
    You are a professional translation evaluator. You must assess a Filipino translation based on:
    - Adequacy: Does the Filipino translation preserve the meaning of the original sentence?.
    - Fluency: Is it natural, smooth, and grammatically correct to be easily understood by a native speaker?.
    - Lexical Choice: Are the words contextually accurate and culturally appropriate?.

    For each input: 
    - Adequacy rating (1-5) + detailed reasoning for your score (cite words or phrases from the translation),
    - Fluency rating (1-5) + reasoning,
    - Lexical Choice rating (1-5) + reasoning.
    - Overall rating (1-5).

    All the reasonings should be detailed.
    Output Format:
    English Sentence: ...
    Filipino Translation: ...
    Adequacy: (1-5) + [reasoning citing specific words/phrases]
    Fluency: (1-5) + [reasoning citing specific words/phrases]
    Lexical Choice: (1-5) + [reasoning citing specific words/phrases]
    Overall Rating: ...

    Do not preamble.
    """

    system_prompt = SystemMessagePromptTemplate.from_template(system_message)

    human_prompt = HumanMessagePromptTemplate.from_template(
        "English Sentence: {english}\nFilipino Translation: {filipino}"
    )

    final_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        human_prompt
    ])

    return final_prompt


def prepare_structured_zeroshot_prompt():
    """Prepare zero-shot prompt with structured JSON output"""
    parser = PydanticOutputParser(pydantic_object=TranslationEvaluation)

    system_message = f"""
    You are a professional translation evaluator. You must assess a Filipino translation based on:
    - Adequacy: Does the Filipino translation preserve the meaning of the original sentence?
    - Fluency: Is it natural, smooth, and grammatically correct to be easily understood by a native speaker?
    - Lexical Choice: Are the words contextually accurate and culturally appropriate?

    CRITICAL: You MUST respond with ONLY valid JSON in this format:
    "overall_score": 4,
    "adequacy_score": 4,
    "fluency_score": 5,
    "lexical_choice_score": 3,
    "adequacy_explanation": "detailed explanation citing specific words/phrases",
    "fluency_explanation": "detailed explanation citing specific words/phrases",
    "lexical_choice_explanation": "detailed explanation citing specific words/phrases",
    "overall_explanation": "summary explanation"
    
    DO NOT include any text before or after the JSON.
    DO NOT use markdown code blocks.
    DO NOT include explanations outside the JSON.
    RESPOND ONLY WITH VALID JSON.

    Be extremely detailed in your explanations and cite specific words or phrases from the translation.
    """

    system_prompt = SystemMessagePromptTemplate.from_template(system_message)

    human_prompt = HumanMessagePromptTemplate.from_template(
        "English Sentence: {english}\nFilipino Translation: {filipino}"
    )

    final_prompt = ChatPromptTemplate.from_messages([
        system_prompt,
        human_prompt
    ])

    return final_prompt


def structured_evaluate_with_prompt_engineering(llm, english: str, filipino: str) -> Dict[str, Any]:
    """
    Simplified structured evaluation using basic prompt engineering (no agents/tools)
    
    Args:
        llm: Language model instance
        english (str): English sentence
        filipino (str): Filipino translation to be evaluated
        
    Returns:
        Dict[str, Any]: Structured evaluation results in JSON format
    """
    
    # Get the structured prompt and parser
    prompt = prepare_structured_zeroshot_prompt()
    
    # Execute the evaluation
    try:
        messages = prompt.format_messages(english=english, filipino=filipino)
        response = llm.invoke(messages)
        
        # Parse the structured output
        cleaned_output = clean_json_response(response.content)
        
        try:
            # Try to parse as JSON first
            structured_result = json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            # Fallback to regex parsing
            structured_result = fallback_parse_evaluation(response.content)
        
        # Ensure all required fields are present
        structured_result = validate_and_complete_output(structured_result, english, filipino)
        
        # Add metadata (matching your format)
        structured_result['metadata'] = {
            'english_sentence': english,
            'filipino_translation': filipino,
            'evaluation_timestamp': str(pd.Timestamp.now()),
            'model_used': str(type(llm).__name__),
        }
        
        return structured_result
        
    except Exception as e:
        # Return error structure if evaluation fails
        return create_error_output(english, filipino, str(e))


def evaluate_with_prompt_engineering(llm, prompt, english: str, filipino: str) -> str:
    """Original evaluation function (for backward compatibility)"""
    messages = prompt.format_messages(english=english, filipino=filipino)
    resp = llm.invoke(messages)
    return resp.content


def fallback_parse_evaluation(response_text: str) -> Dict[str, Any]:
    """Simplified fallback parser for basic prompt output"""
    
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
            r"adequacy\s*:\s*\d+\s*[-+]?\s*(.*?)(?=\nfluency|\nlexical|\noverall|$)", 
            response_text
        ),
        "fluency_explanation": extract_explanation(
            r"fluency\s*:\s*\d+\s*[-+]?\s*(.*?)(?=\nlexical|\noverall|$)", 
            response_text
        ),
        "lexical_choice_explanation": extract_explanation(
            r"lexical\s+choice\s*:\s*\d+\s*[-+]?\s*(.*?)(?=\noverall|$)", 
            response_text
        ),
        "overall_explanation": extract_explanation(
            r"overall\s+rating\s*:\s*(\d+)", 
            response_text
        )
    }

def validate_and_complete_output(result: Dict[str, Any], english: str, filipino: str) -> Dict[str, Any]:
    """Ensure all required fields are present and valid"""
    
    # Required fields with defaults (using 0 as you specified)
    required_fields = {
        "overall_score": 0,
        "adequacy_score": 0,
        "fluency_score": 0,
        "lexical_choice_score": 0,
        "adequacy_explanation": "No explanation provided",
        "fluency_explanation": "No explanation provided", 
        "lexical_choice_explanation": "No explanation provided",
        "overall_explanation": "No overall explanation provided"
    }
    
    # Fill missing fields
    for field, default in required_fields.items():
        if field not in result or result[field] is None or result[field] == "":
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
        "adequacy_explanation": f"Evaluation failed : {error_message}",
        "fluency_explanation": f"Evaluation failed: {error_message}",
        "lexical_choice_explanation": f"Evaluation failed: {error_message}",
        "overall_explanation": f"Evaluation failed: {error_message}",
        "metadata": {
            "english_sentence": english,
            "filipino_translation": filipino,
            "evaluation_timestamp": str(pd.Timestamp.now()),
            "error": True,
            "error_message": error_message,
        }
    }

def clean_json_response(response_content):
    """Clean LLM output for JSON parsing"""
    
    # Strip leading/trailing whitespace
    text = response_content.strip()
    
    # Remove markdown code block fences
    if text.startswith("```json"):
        text = re.sub(r"^```json\s*", "", text)
    if text.startswith("```"):
        text = re.sub(r"^```\s*", "", text)
    if text.endswith("```"):
        text = re.sub(r"\s*```$", "", text)

    # Remove any text before the first {
    if '{' in text:
        start_idx = text.find('{')
        text = text[start_idx:]
    
    # Remove any text after the last }
    if '}' in text:
        end_idx = text.rfind('}') + 1
        text = text[:end_idx]

    # Remove trailing commas
    text = re.sub(r",\s*([}\]])", r"\1", text)

    return text.strip()

def measure_standard_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )

    print(f"Accuracy: {acc:.3f}")
    print(f"Macro Precision: {prec:.3f}")
    print(f"Macro Recall: {rec:.3f}")
    print(f"Macro F1-score: {f1:.3f}")

    print("\nDetailed per-rating breakdown:\n")
    print(classification_report(y_true, y_pred))

def parse_rating(text):
    """Original rating parser (for backward compatibility)"""
    m = re.search(r"Rating:\s*([1-5])", text)
    return int(m.group(1)) if m else None