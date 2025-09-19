Damian AI: A Personalized, Knowledge-Based Conversational Agent
Damian AI is a sophisticated digital persona designed to embody the knowledge and unique cognitive style of its creator. It is a conversational interface into a complete body of public work, architected to reason and communicate using the same systematic, logic-driven models detailed in the source articles.

This project is a successful proof-of-concept demonstrating how to solve complex AI challenges like long-term memory, persona drift, and contextual synthesis on an efficient, locally-run Large Language Model.

The complete code is provided here so that developers, researchers, and potential collaborators can run, inspect, and build upon this architecture.

Key Architectural Features
This is not a simple RAG chatbot. The Damian AI runs on an advanced cognitive architecture designed for high-fidelity persona embodiment.

Curated Knowledge Base: The AI's knowledge is sourced exclusively from a local database.py file. This eliminates the reliability issues of web scraping and provides absolute control over the AI's factual grounding.

Immutable Constitution: The AI's core identity is defined in damian_constitution.md. This "prime directive" is a non-negotiable set of rules that governs its tone, logic, and communication style, ensuring the persona remains stable.

"Decision-Execution" Cognitive Model: To prevent the persona failures common in monolithic prompts, Damian AI uses a two-step cognitive process inspired by the "Jeremy" AI architecture.

Decision Layer: A meta-mind first analyzes the user's query to classify its intent (e.g., a direct question vs. a multi-part query requiring synthesis).

Execution Layer: Based on that decision, it calls a specialized function with a hyper-focused prompt, allowing the LLM to execute the task flawlessly without deviating from its persona.

Local LLM Powered: The entire system is designed to run with a local LLM (e.g., Llama 3 8B via LM Studio or Ollama). This prioritizes data privacy, eliminates API dependencies, and provides instant response times.

A Note on Public Hosting
This application is designed to be run locally. Due to the resource requirements of the core AI libraries (specifically sentence-transformers for creating embeddings), hosting this application on free cloud platforms is not feasible as they typically lack the necessary RAM, leading to server crashes.

The code is provided here for those who wish to run it on their personal computers and explore its architecture.

How to Run This Project Locally
To run Damian AI on your own computer, you will need Python 3.9+ and a local LLM server that provides an OpenAI-compatible API endpoint.

Prerequisites
Python: Ensure Python and Pip are installed.

Local LLM Server: You must have a local server running, such as LM Studio or Ollama. The server must be configured to provide an API endpoint at http://127.0.0.1:1234/v1.

Step-by-Step Instructions
Clone the Repository:

git clone [https://github.com/your-username/damian-ai.git](https://github.com/your-username/damian-ai.git)
cd damian-ai

Install Dependencies:
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

Customize the Knowledge (Optional):
Open the database.py file. You can replace the existing content with your own articles and texts to create a personalized AI based on a different body of work.

Customize the Persona (Optional):
Open the damian_constitution.md file to edit the core principles, communication style, and identity of the AI.

Run the Application:
Ensure your local LLM server is running. Then, launch the Streamlit application from your terminal:

streamlit run app.py
