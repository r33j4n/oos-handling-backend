from langchain_community.llms import Bedrock
from langchain_aws import BedrockLLM
from langchain_community.llms.ollama import Ollama


mitsral_large= "mistral.mistral-large-2402-v1:0"
meta_llama3_70b="meta.llama3-70b-instruct-v1:0"

def get_bedrock_model():
    """
    Creates and returns a LangChain LLMChain using the Llama 2 70B chat model from Bedrock.

    Returns:
        LLMChain: An LLMChain object configured with the Llama 2 model.
    """

    # Initialize the Bedrock model
    bedrock_client = BedrockLLM(
        model_id= meta_llama3_70b,
        region_name="us-east-1",  # Choose the appropriate region
        credentials_profile_name="default"  # Use your default AWS profile or specify a different one
    )

    # Create a LangChain LLMChain
    return bedrock_client
# Import Ollama Model

def get_ollama_model():
    """
    Creates and returns an instance of the Ollama model.

    Returns:
        Ollama: An Ollama model object.
    """
    # Initialize the Ollama model
    ollama_client = Ollama(
        model_id="ollama.ollama-large-2402-v1:0",  # Replace with the appropriate model ID
        region_name="us-east-1",  # Choose the appropriate region
        credentials_profile_name="default"  # Use your default AWS profile or specify a different one
    )

    return ollama_client
