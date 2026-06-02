"""
Evaluation Support Functions
"""
from ragas import EvaluationDataset, SingleTurnSample

def retrieve_docs(question):
    return retriever.invoke(question)

def build_ragas_dataset(samples):
    ragas_samples = []

    for sample in samples:
        ragas_samples.append(
            SingleTurnSample(
                user_input=sample["question"],
                response=sample["answer"],
                retrieved_contexts=sample["contexts"],
                reference="NO_REFERENCE",
            )
        )

    return EvaluationDataset(samples=ragas_samples)
