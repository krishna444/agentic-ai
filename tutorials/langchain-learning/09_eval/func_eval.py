from langsmith import Client, evaluate, traceable
from inventory_agent import run
from langchain.tools import tool
from utils import cosine_similarity 
from dotenv import load_dotenv
load_dotenv()

@traceable
def target(inputs: dict) -> dict:
    
    question = inputs["question"]
    answer = run(question)
    return {"answer": answer}

client = Client()

dataset_name = "inventorydata"

if not client.has_dataset(dataset_name=dataset_name):
    client.create_dataset(dataset_name=dataset_name)
    client.cleanup()
    """
    client.create_examples(
        dataset_name=dataset_name,
        examples=[
            {
                "inputs": {"question": "What is the stock status of iPhone 15?"},
                "outputs": {"answer": "The iPhone 15 is currently in stock with 2 units available."},
            },
            {
                "inputs": {"question": "Is AirPods Pro available?"},
                "outputs": {"answer": "The AirPods Pro is currently out of stock. There are 0 available items."},
            },
            {
                "inputs": {"question": "How many iPhone 15 units are available?"},
                "outputs": {"answer": "The iPhone 15 is currently in stock with 2 units available."},
            },            
            {
                "inputs": {"question": "What is the capital city of Nepal?"},
                "outputs": {"answer": "The capital city of Nepal is Kathmandu."},
            },
            {
                "inputs": {"question": "What is the battery life of the MacBook Pro?"},
                "outputs": {"answer": "The MacBook Pro offers up to 20 hours of battery life."}
            }
        ],
    )
    """

def semantic_match(example, run):
    expected = example.outputs["answer"]
    actual = run.outputs["answer"]
    sim = cosine_similarity(expected, actual)
    return {
        "key": "semantic_match",
        "score": float(sim)
    }

evaluate(
    target,
    client=client,
    data=dataset_name,
    evaluators=[semantic_match],
    experiment_prefix="inventory_agent_evaluation_qwen3-32b"
)