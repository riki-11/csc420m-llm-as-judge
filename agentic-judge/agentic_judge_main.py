from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from tqdm import tqdm
from langchain_chroma import Chroma

import requests
import json
import re


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
    print("Closest training examples (based on English input):")
    for doc in nearest_docs:
        content = doc.page_content
        metadata = doc.metadata

        print(f"- {doc.page_content}")
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