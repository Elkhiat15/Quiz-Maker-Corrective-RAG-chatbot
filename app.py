import streamlit as st
from parsers.doc import DocumentExtractor
from utils.helper import set_env_keys
from database.chromadb import VectorDB
from agent_graph.graph import compile_workflow
from constants.constansts import OUTPUT_FOLDER
import shutil
import os

set_env_keys()

file = st.file_uploader("Upload file", type=["pdf", "pptx"])
if file:
    extractor = DocumentExtractor()
    extractor.extract(file)
    documents = extractor.get_documents()
    vector_db = VectorDB(documents)
    retriever = vector_db.get_retriever()
    app = compile_workflow()
    # Run
    inputs = {
        "question": "How to list files in current directory using long format?",
        "retriever": retriever
    }
    for output in app.stream(inputs):
        for key, value in output.items():
            pass
            # st.write(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        # st.write("\n---\n")

    # Final generation
    st.markdown(value["generation"])
    if value["relevant_images"]:
        st.image(value["relevant_images"][0], caption=None)
    # st.write(documents)
else:
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)
