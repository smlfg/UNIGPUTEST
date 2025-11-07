"""
Prompt Templates for RAG
"""

from typing import List, Optional


class PromptTemplate:
    """Base prompt template"""

    def __init__(self, template: str):
        """
        Initialize prompt template

        Args:
            template: Template string with {placeholders}
        """
        self.template = template

    def format(self, **kwargs) -> str:
        """
        Format template with variables

        Args:
            **kwargs: Variables to substitute

        Returns:
            Formatted prompt
        """
        return self.template.format(**kwargs)


class RAGPromptBuilder:
    """
    Build prompts for RAG

    Combines query and retrieved context into effective prompts
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant. Your task is to answer user questions based on the provided context.

Instructions:
- Use ONLY the information from the provided context to answer the question
- If the context doesn't contain relevant information, say "I don't have enough information to answer this question"
- Be concise and accurate
- Cite specific parts of the context when relevant
- Do not make up information not present in the context"""

    MISTRAL_TEMPLATE = """<s>[INST] {system_prompt}

Context:
{context}

Question: {query} [/INST]"""

    LLAMA_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

Context:
{context}

Question: {query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with context that provides further information. Write a response that appropriately answers the question.

### System:
{system_prompt}

### Context:
{context}

### Question:
{query}

### Response:
"""

    def __init__(
        self,
        template_type: str = "mistral",
        system_prompt: Optional[str] = None
    ):
        """
        Initialize prompt builder

        Args:
            template_type: Type of template ('mistral', 'llama', 'alpaca')
            system_prompt: Custom system prompt (optional)
        """
        self.template_type = template_type
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Select template
        if template_type == "mistral":
            self.template = self.MISTRAL_TEMPLATE
        elif template_type == "llama":
            self.template = self.LLAMA_TEMPLATE
        elif template_type == "alpaca":
            self.template = self.ALPACA_TEMPLATE
        else:
            raise ValueError(f"Unknown template type: {template_type}")

    def build_prompt(
        self,
        query: str,
        context: List[str],
        max_context_chunks: int = 5
    ) -> str:
        """
        Build RAG prompt

        Args:
            query: User query
            context: List of context chunks
            max_context_chunks: Maximum number of context chunks to include

        Returns:
            Formatted prompt
        """
        # Limit context chunks
        context_chunks = context[:max_context_chunks]

        # Format context
        context_str = self._format_context(context_chunks)

        # Build prompt
        prompt = self.template.format(
            system_prompt=self.system_prompt,
            context=context_str,
            query=query
        )

        return prompt

    def _format_context(self, context_chunks: List[str]) -> str:
        """
        Format context chunks

        Args:
            context_chunks: List of context strings

        Returns:
            Formatted context string
        """
        formatted_chunks = []

        for i, chunk in enumerate(context_chunks, 1):
            formatted_chunks.append(f"[{i}] {chunk}")

        return "\n\n".join(formatted_chunks)

    def build_simple_prompt(self, query: str) -> str:
        """
        Build simple prompt without context (for testing)

        Args:
            query: User query

        Returns:
            Formatted prompt
        """
        if self.template_type == "mistral":
            return f"<s>[INST] {query} [/INST]"
        elif self.template_type == "llama":
            return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            return f"Question: {query}\n\nAnswer:"


# Predefined prompts for common RAG scenarios

SUMMARIZATION_PROMPT = """You are a summarization assistant. Summarize the provided context to answer the user's question.

Focus on:
- Key points relevant to the question
- Main ideas and conclusions
- Concrete facts and figures

Be concise and informative."""

QA_PROMPT = """You are a question-answering assistant. Answer the user's question using only the provided context.

Guidelines:
- Directly answer the question
- Use exact quotes when helpful
- If uncertain, say so
- Don't add information not in the context"""

CODE_PROMPT = """You are a code documentation assistant. Help users understand code based on the provided context.

When answering:
- Explain code functionality clearly
- Highlight important patterns
- Provide examples if relevant
- Reference specific functions/classes mentioned"""

RESEARCH_PROMPT = """You are a research assistant. Help users understand complex topics based on the provided context.

Your responses should:
- Explain concepts clearly
- Connect related ideas
- Highlight important findings
- Acknowledge limitations in the context"""
