import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()  # Ensures .env is loaded

from agent.router_orchestrator import RecommendationAgent

def test_direct_request():
    print("Initializing router orchestrator with local Ollama...")
    agent = RecommendationAgent()
    
    print("\nOrchestrator ready! Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("> ")
            if query.lower() in ['exit', 'quit']:
                break
            if not query.strip():
                continue
            
            response = agent.run(query)
            print(f"Assistant: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    test_direct_request()
