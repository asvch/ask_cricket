# run this file: streamlit run .\evaluate_rag.py

import os
from ragas import evaluate  # Uses OpenAI API 
from ragas.metrics import (
    faithfulness,
    answer_relevancy,          
    answer_correctness,
    context_recall,
    context_precision,
)
from datasets import Dataset

# import chain from other file
from ask_anything_cricket import chain

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

eval_questions = [  # Ground truth setup required for some metrics
        "How many players are normally in each team in a game?",
        "How do the umpires signal a wide ball?",
        "What are the dimensions of the pitch in meters?",
        "Who is the most responsible for ensuring the game is played in the right spirit?",
        "How many individual stumps are on a cricket pitch during a game?"
    ]

eval_answers = [   # answers do not need to be exactly this
        "11 players play in each team in a game.",
        "The umpires signal a wide by extending both arms horizontally.",
        "The pitch is a rectangular area of the ground 20.12 m in length and 3.05 m in width.",
        "The captains are most responsible for ensuring the spirit of fair play.",
        "There are six individual stumps on a cricket pitch during a game."
    ]

# Iterate thru each eval question & collect answers & contexts
answers = []    # Ragas needs this info to compute the metrics
contexts = []

for question in eval_questions:
    response = chain.invoke({"input": question})
    answers.append(response["answer"])
    contexts.append([context.page_content for context in response["context"]])


# We must massage the results into Hugging Face format for Ragas.
response_dataset = Dataset.from_dict({
    "question" : eval_questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth" : eval_answers
})

# Tell Ragas which metrics we're interested in
metrics = [
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
]

# call evaluate w/ dataset created above & metrics we want to measure
ragas_results = evaluate(response_dataset, metrics)  # uses OpenAI API as per ragas docs
print(ragas_results)

df = ragas_results.to_pandas()

# Save the dataframe to a CSV file
df.to_csv('evaluation_results.csv', index=False)
