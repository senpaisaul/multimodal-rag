import matplotlib.pyplot as plt
import json
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


class GraphAgent:
    def __init__(self, groq_key: str):
        self.llm = ChatGroq(
            api_key=groq_key,
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    def generate(self, context: str, query: str):
        prompt = ChatPromptTemplate.from_template("""
Return STRICT JSON only.

{
 "graph_type": "line|bar|scatter",
 "title": "",
 "x_label": "",
 "y_label": "",
 "data_points": [[x,y]]
}

Context:
{context}

Query:
{query}
""")

        raw = (prompt | self.llm).invoke({
            "context": context,
            "query": query
        }).content

        spec = json.loads(raw)

        if not spec.get("data_points"):
            raise ValueError("No data points available for graph")

        xs, ys = zip(*spec["data_points"])

        fig, ax = plt.subplots()
        if spec["graph_type"] == "line":
            ax.plot(xs, ys)
        else:
            ax.bar(xs, ys)

        ax.set_title(spec.get("title", ""))
        ax.set_xlabel(spec.get("x_label", ""))
        ax.set_ylabel(spec.get("y_label", ""))

        return fig

