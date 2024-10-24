from langchain.schema import Document
import streamlit as st
from models.models import (
    get_retrieval_grader,
    get_rag_chain,
    get_question_rewriter,
    get_web_search_tool
)


def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    with st.spinner("Retrieving ..."):
        question = state["question"]

        # Retrieval
        # vectorstore = Chroma(
        #     collection_name=COLLECTION_NAME,  # Name of the collection
        #     persist_directory=PERSIST_DIRECTORY,
        #     embedding_function=MistralAIEmbeddings(model="mistral-embed")
        # )
        # retriever = vectorstore.as_retriever()
        retriever = state['retriever']
        documents = retriever.get_relevant_documents(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    with st.spinner("Generating ..."):
        question = state["question"]
        documents = state["documents"]
        relevant_images = []
        for d in documents:
            if d.metadata['type'] == 'image':
                relevant_images.append(d.metadata['img_path'])
        rag_chain = get_rag_chain()
        generation = rag_chain.invoke({"context": documents, "question": question})

    return {
        "documents": documents,
        "question": question,
        "generation": generation,
        "relevant_images": relevant_images}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    with st.spinner("Checking document relevance ..."):
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            retrieval_grader = get_retrieval_grader()
            score = retrieval_grader.invoke({"question": question, "document": d.page_content})
            grade = "no"
            if score is not None and hasattr(score, 'binary_score'):
                grade = score.binary_score
            if grade == "yes":
                # print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                # print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def transform_query(state):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """

    with st.spinner("Transforming the query ..."):
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        question_rewriter = get_question_rewriter()
        better_question = question_rewriter.invoke({"question": question})
    return {"documents": documents, "question": better_question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    # print("---WEB SEARCH---")
    with st.spinner("Searching the web ..."):
        question = state["question"]
        documents = state["documents"]

        # Web search
        web_search_tool = get_web_search_tool(k=3)
        docs = web_search_tool.invoke({"query": question})
        web_results = "\n".join([f"Content from Url:{d['url']}:\n{d['content']}" for d in docs])
        web_results = Document(
            page_content=web_results,
            metadata={
                'type': 'text'
            })
        documents.append(web_results)

    return {"documents": documents, "question": question}


def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    # print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        # print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        # print("---DECISION: GENERATE---")
        return "generate"
