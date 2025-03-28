from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
# from services.sentiment import SentimentAnalyzer  # Assume you have this
from controllers.qa_handler import QAHandler  # Your existing QA handler
from services.google_ai import GoogleAI
import operator
from langchain.prompts import ChatPromptTemplate

class BotState(TypedDict):
    user_input: str
    sentiment: Annotated[str, "negative|positive"]
    qa_answer: str
    needs_human: bool
    response: str

class CustomerSupportBot:
    def __init__(self):
        self.llm = GoogleAI().get_llm()
        self.qa_handler = QAHandler()
        self.workflow = StateGraph(BotState)
        self.retriever = None
        
        # Define nodes
        self.workflow.add_node("analyze_sentiment", self.analyze_sentiment)
        self.workflow.add_node("retrieve_answer", self.retrieve_answer)
        self.workflow.add_node("escalate_to_human", self.escalate_to_human)
        self.workflow.add_node("generate_response", self.generate_response)

        # Define edges
        self.workflow.set_entry_point("analyze_sentiment")
        self.workflow.add_edge("generate_response", END)
        
        # Conditional routing
        self.workflow.add_conditional_edges(
            "analyze_sentiment",
            self.decide_routing,
            {
                "escalate": "escalate_to_human",
                "answer": "retrieve_answer"
            }
        )
        self.workflow.add_edge("retrieve_answer", "generate_response")
        self.workflow.add_edge("escalate_to_human", "generate_response")

        # Compile the graph
        self.app = self.workflow.compile()
    
    def sentiment_analyzer(self, text: str) -> str:
        """Analyze sentiment of the text"""
        prompt = ChatPromptTemplate.from_template(
            """
        You are a customer support chatbot. Analyze the sentiment of the following customer query. 
        Ignore any previous chat history. Focus solely on the current query.
        Consider the context of customer support interactions when determining the sentiment.

        Respond with one of the following exactly: 'positive', or 'negative'.

        Here are some guidelines:

        - 'positive': The customer expresses satisfaction, appreciation, or positive feelings.The customer's query is informational, factual, or lacks strong emotional expression, and the chatbot can potentially provide a resolution.
            If the customer uses inappropriate language, but the query can still be answered by the bot, respond with neutral.
        - 'negative': The customer explicitly requests to speak to a human, or the customer's query indicates that the chatbot is unable to provide a satisfactory answer and requires human intervention.

        Query: {query}
        """
        )
        chain = prompt | self.llm
        sentiment = chain.invoke({"query": text}).content
        return sentiment

    def analyze_sentiment(self, state: BotState) -> dict:
        """Analyze user input sentiment"""
        sentiment = self.sentiment_analyzer(state["user_input"])
        return {"sentiment": sentiment}

    def decide_routing(self, state: BotState) -> str:
        """Route based on sentiment"""
        if state["sentiment"] == "negative":
            return "escalate"
        return "answer"

    def retrieve_answer(self, state: BotState) -> dict:
        """Get answer from knowledge base"""
        answer = self.qa_handler.get_answer(state["user_input"], self.retriever)
        return {"qa_answer": answer}

    def escalate_to_human(self, state: BotState) -> dict:
        """Handle escalation"""
        return {"needs_human": True}

    def generate_response(self, state: BotState) -> dict:
        """Final response formatting"""
        if state.get("needs_human"):
            return {"response": "Let me connect you to a human representative..."}
        return {"response": state["qa_answer"]}

    def handle_query(self, user_input: str, retriever) -> str:
        """Execute the workflow"""
        self.retriever = retriever
        results = self.app.invoke({"user_input": user_input})
        return results["response"]