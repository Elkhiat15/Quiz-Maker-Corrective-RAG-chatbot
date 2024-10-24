description_prompt = """
Analyze the following photo from an academic document and provide a concise description,
including relevant details that would aid in retrieving the photo when questioned about its contents.
Exclude any extraneous text. Use the following context around the image:  
{context}
If the image seems to be useless (e.g., logos or personal photos), respond with (no description) and indicate it is not useful.
{format_instructions}
"""

grader_prompt_template = """
You are a grader assessing relevance of a retrieved document to a user question.   
If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.   
Provide a binary score: 'yes' if the document is relevant, and 'no' if it is not.  
Retrieved document: \n\n {document} \n\n User question: {question}
"""

rewriter_system_prompt = """
You a question re-writer that converts an input question to a better version that is optimized for web search.
Look at the input and try to reason about the underlying sematic intent / meaning.
Give the question only without any acknowledgement or extra words.
"""

rag_prompt_template = """human

You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
Keep the answer concise.
Show the resources where the content come from, the resources may be URLs OR page numbers. 
Please include the sources of the information, which can be URLs or page numbers. 

IGNORE image paths and page content, just show image number if found.

Question: {question} 

Context: {context} 

Answer: 

"""
