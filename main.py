import os
import pandas as pd
import numpy as np
from owlready2 import get_ontology
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.runnables import RunnablePassthrough
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer, util


def load_ontology(file_path):
    """Loads the OWL ontology from the specified file path."""
    print("Loading ontology...")
    return get_ontology("file://" + file_path).load()


def load_pdf_data(file_path):
    """Loads and splits the content of a PDF file into chunks."""
    if file_path:
        loader = UnstructuredPDFLoader(file_path=file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7500, chunk_overlap=100
        )
        return text_splitter.split_documents(data)
    else:
        raise FileNotFoundError("Upload a PDF file")


def setup_qa_chain(chunks, local_model="mistral"):
    """Sets up the question-answering chain using Ollama embeddings and LLM."""
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
        collection_name="local-rag",
    )

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""answer_prompt_template = You are an expert in electrial engineering.
        Using only the information provided in the context, choose the best answer, and only that, to the following question:
        Original question: {question}""",
    )

    llm = ChatOllama(model=local_model)
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )

    template = """Answer the question based ONLY on the following context with ONLY one answer:
    {context}
    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def update_ontology(onto, individuals_to_update, chain, file_path):
    """Iterates over individuals, queries the context, and updates the ontology.

    Args:
        onto: The loaded owlready2 ontology object.
        individuals_to_update: A list of individuals to be updated.
        chain: The Langchain QA chain for generating answers.
        file_path: The path to the OWL file where updates will be saved.
    """
    with onto:
        for individual in individuals_to_update:
            try:
                if not individual:
                    print(f"Warning: Individual with IRI {individual.iri} not found.")
                    continue

                term_lexicon_string_value = None
                if hasattr(individual, "termLexiconString"):
                    term_lexicon_string_value = getattr(individual, "termLexiconString")
                    if isinstance(term_lexicon_string_value, list):
                        term_lexicon_string_value = " ".join(term_lexicon_string_value)

                if term_lexicon_string_value and term_lexicon_string_value.strip():
                    lexicon_question = "What is " + term_lexicon_string_value + "?"

                    if (
                        hasattr(individual, "termMeaningString")
                        and individual.termMeaningString
                    ):
                        print(
                            f"termMeaningString already exists for {term_lexicon_string_value}. Skipping."
                        )
                    else:
                        answer = chain.invoke(lexicon_question)
                        print("Question:", lexicon_question)
                        print(term_lexicon_string_value, ":", answer)
                        individual.termMeaningString.append(answer)
                else:
                    print(
                        f"Warning: termLexiconString is empty or not found for individual {individual.iri}"
                    )

            except Exception as e:
                print(f"Error processing individual {individual.iri}: {e}")

    # Save the updated ontology (only once after ALL updates are done)
    onto.save(file_path)


def main():
    # Define file paths (adjust as needed)
    directory = "D:\GitHub\Projetos\Mestrado\EnergyContext\ontology\imports"
    filename = "oec-extracted.owl"
    file_path = os.path.join(directory, filename)
    default_save_path = r"D:\GitHub\Projetos\Mestrado\EnergyContext\ontology\imports\updated_oec-extracted.owl"
    pdf_path = r"D:\GitHub\Projetos\Mestrado\EnergyContext\pdf\appendixa_0.pdf"

    # Load ontology
    onto = load_ontology(file_path)

    # Define IRIs
    term_sent_by_property_iri = (
        "http://www.semanticweb.org/matheus/ontologies/2023/10/oec-extracted#termSentBy"
    )
    actor_coal_iri = (
        "http://www.semanticweb.org/matheus/ontologies/2023/10/oec-extracted#actorCoal"
    )
    actor_has_context_property_iri = "http://www.semanticweb.org/matheus/ontologies/2023/10/oec-extracted#actorHasContext"
    # Get ontology elements
    term_sent_by_property = onto.search_one(iri=term_sent_by_property_iri)
    actor_coal = onto.search_one(iri=actor_coal_iri)
    actor_has_context_property = onto.search_one(iri=actor_has_context_property_iri)

    # Load and split PDF
    chunks = load_pdf_data(pdf_path)

    # Setup QA chain
    chain = setup_qa_chain(chunks)

    # List to store individuals to be updated
    individuals_to_update = []

    # Collect individuals to update
    if term_sent_by_property and actor_coal and actor_has_context_property:
        for individual in onto.individuals():
            if term_sent_by_property in individual.get_properties():
                term_sent_by_values = getattr(
                    individual, term_sent_by_property.python_name
                )
                if actor_coal in term_sent_by_values:
                    if hasattr(individual, "termLexiconString"):
                        if not isinstance(individual.termLexiconString, list):
                            individual.termLexiconString = [
                                individual.termLexiconString
                            ]
                    individuals_to_update.append(individual)

    # Update ontology (Make sure to include file_path in this call)
    update_ontology(onto, individuals_to_update, chain, file_path)
    print("finished")


if __name__ == "__main__":
    main()
