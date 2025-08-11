from langchain_core.prompts import ChatPromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

import pandas as pd
import re

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


def evaluate_with_prompt_engineering(llm, prompt, english: str, filipino: str) -> str:
    messages = prompt.format_messages(english=english, filipino=filipino)
    resp = llm.invoke(messages)
    return resp.content


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
    m = re.search(r"Rating:\s*([1-5])", text)
    return int(m.group(1)) if m else None