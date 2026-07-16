import requests

class DebateConductor:
    def __init__(self):
        # We enforce local Ollama default port for maximum data privacy
        self.ollama_url = "http://localhost:11434/api/generate"
        self.debate_history = []
        # INTERN TASK: Select a lightweight model like 'mistral' or 'llama3'
        self.model = "mistral"

    def generate_agent_a_response(self, topic):
        """
        Agent A fiercely DEFENDS the topic.
        """
        # --- INTERN CODE GOES HERE ---
        # 1. Build the System Prompt assigning a specific persona.
        # 2. Extract self.debate_history and format it into the prompt.
        # 3. Use requests.post() to send the payload to self.ollama_url
        # 4. Parse the JSON response stream from Ollama.
        
        return "Not Implemented! Connect me to Ollama!"

    def generate_agent_b_response(self, topic):
        """
        Agent B fiercely CHALLENGES the topic.
        """
        # --- INTERN CODE GOES HERE ---
        # 1. Build the System Prompt.
        # 2. Execute the REST POST request to Ollama locally.
        
        return "Not Implemented! Connect me to Ollama!"
