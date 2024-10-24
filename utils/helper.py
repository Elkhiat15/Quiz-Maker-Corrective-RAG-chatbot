from langchain_core.messages import HumanMessage
from langchain.output_parsers import PydanticOutputParser
from chromadb import PersistentClient
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from prompts.prompts import description_prompt
import base64
import os


def set_env_keys():
    load_dotenv()
    # os.environ['LANGCHAIN_TRACING_V2'] = 'true'
    # os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'


def get_img_data(image_data):
    image_data_b64 = base64.b64encode(image_data).decode("utf-8")
    return image_data_b64


def get_description(img_data , context: str):
    class ImageAnalysis(BaseModel):
        description: str = Field(description="Description of the photo")
        useful: str = Field(description="Indicates if the photo is useful for retrieval (yes or no)")

    parser = PydanticOutputParser(pydantic_object=ImageAnalysis)

    # Get the base64-encoded image data
    image_data_b64 = get_img_data(img_data)

    # Determine the image format
    mime_type = 'image/png' # if img_path.lower().endswith('.png') else 'image/jpeg'

    # Create a HumanMessage object with the prompt and image
    input_message = HumanMessage(
        content=[
            {"type": "text", "text": description_prompt.format(context = context, format_instructions=parser.get_format_instructions())},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{image_data_b64}"},
            },
        ],
    )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", max_tokens=1024)
    response = llm.invoke([input_message])
    # Parse the response using the Pydantic parser
    parsed_response = parser.parse(response.content)

    return parsed_response.description, parsed_response.useful


def onerror(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def delete_collection(path):
    try:
        collection_name = path
        chroma_client = PersistentClient(path=".chroma")
        chroma_client.delete_collection(collection_name)
        print(f"Collection {collection_name} deleted successfully.")
    except Exception as e:
        raise Exception(f"Unable to delete collection: {e}")
