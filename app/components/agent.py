from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import logging
from ..utils import genai_client

@dataclass
class RetrievalAssessment:
    is_sufficient: bool
    reasoning: str
    needs_clarification: bool
    suggested_clarification: Optional[str] = None

@dataclass
class QueryRewrite:
    original_query: str
    rewritten_query: str
    reasoning: str

class Agent:
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        max_retries: int = 3,
        temperature: float = 0.2
    ):
        """Initialize the Agent with configuration."""
        self.max_retries = max_retries
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)

    async def assess_retrieval(self, query: str, retrieved_chunks: List[str]) -> RetrievalAssessment:
        """
        Assess if retrieved chunks are sufficient to answer the query.
        Returns assessment with reasoning and potential clarification needs.
        """
        prompt = f"""You are an assistant evaluating whether a set of retrieved text chunks sufficiently answer a user's original question.

User Query: {query}

Retrieved Context Chunks:
{'-' * 40}
{chr(10).join(retrieved_chunks)}
{'-' * 40}

Instructions:
- Your job is to assess whether the provided context chunks are sufficient to answer the user's original question.
- If the context is sufficient, say so.
- If not, suggest a rephrasing of the original question or indicate that the context is insufficient.
- Do NOT ask the user for more information or open new lines of inquiry.
- Do NOT ask clarifying questions to the user.
- In the SUGGESTED_CLARIFICATION field, do NOT ask a question to the user. Instead, if the original question was ambiguous or missing details, suggest a more specific, rephrased version of the original question that would help retrieval. If no rephrase is needed, return 'None'.

Format your response as:
SUFFICIENT: [true/false]
REASONING: [your detailed assessment]
NEEDS_CLARIFICATION: [true/false]
SUGGESTED_CLARIFICATION: [a more specific or rephrased version of the original question if needed, or 'None' if not needed]
"""
        try:
            response = genai_client.generate_content(
                prompt=prompt,
                temperature=self.temperature
            )
            
            # Parse the response
            lines = response.strip().split('\n')
            assessment_dict = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    assessment_dict[key.strip()] = value.strip()

            return RetrievalAssessment(
                is_sufficient=assessment_dict.get('SUFFICIENT', '').lower() == 'true',
                reasoning=assessment_dict.get('REASONING', ''),
                needs_clarification=assessment_dict.get('NEEDS_CLARIFICATION', '').lower() == 'true',
                suggested_clarification=assessment_dict.get('SUGGESTED_CLARIFICATION', 'None') if assessment_dict.get('NEEDS_CLARIFICATION', '').lower() == 'true' else None
            )
        except Exception as e:
            self.logger.error(f"Error in assess_retrieval: {str(e)}")
            return RetrievalAssessment(
                is_sufficient=False,
                reasoning=f"Error during assessment: {str(e)}",
                needs_clarification=False
            )

    async def rewrite_query(self, original_query: str, feedback: Optional[str] = None) -> QueryRewrite:
        """
        Rewrite the query to be more specific or effective based on feedback.
        Returns the rewritten query with reasoning.
        """
        prompt = f"""Task: Improve the user's query to make it more specific and effective for retrieval.

Original Query: {original_query}

{f'Feedback/Context: {feedback}' if feedback else ''}

Analyze the query and rewrite it to:
1. Be more specific and targeted
2. Include key terms that might help in retrieval
3. Maintain the original intent while being more precise

Format your response as:
REWRITTEN_QUERY: [your rewritten query]
REASONING: [explanation of your changes]
"""
        try:
            response = genai_client.generate_content(
                prompt=prompt,
                temperature=self.temperature
            )
            
            # Parse the response
            lines = response.strip().split('\n')
            rewrite_dict = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    rewrite_dict[key.strip()] = value.strip()

            return QueryRewrite(
                original_query=original_query,
                rewritten_query=rewrite_dict.get('REWRITTEN_QUERY', original_query),
                reasoning=rewrite_dict.get('REASONING', '')
            )
        except Exception as e:
            self.logger.error(f"Error in rewrite_query: {str(e)}")
            return QueryRewrite(
                original_query=original_query,
                rewritten_query=original_query,
                reasoning=f"Error during query rewrite: {str(e)}"
            )

    async def decompose_query(self, query: str) -> list:
        """
        Use the LLM to break a complex query into sub-questions for improved retrieval.
        Returns a list of sub-questions (strings).
        """
        prompt = f"""You are an expert at breaking down complex questions into simpler sub-questions for information retrieval.

Original Query: {query}

Instructions:
- If the query can be decomposed into multiple sub-questions that would help retrieve more relevant information, list them.
- If not, just return the original query.
- Output only the sub-questions, one per line, with no numbering or extra text.
"""
        try:
            response = genai_client.generate_content(
                prompt=prompt,
                temperature=self.temperature
            )
            sub_questions = [q.strip() for q in response.strip().split('\n') if q.strip()]
            if not sub_questions:
                return [query]
            return sub_questions
        except Exception as e:
            self.logger.error(f"Error in decompose_query: {str(e)}")
            return [query]

    async def process_query(
        self,
        query: str,
        retrieval_func: callable,
        generation_func: callable,
        schema_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main query processing pipeline with retry logic, assessment, and decomposition.
        Returns the final response along with process metadata.
        """
        retries = 0
        process_log = []
        current_query = query
        while retries <= self.max_retries:
            try:
                process_log.append({
                    "step": "attempt",
                    "retry_count": retries,
                    "query": current_query
                })
                # Retrieve chunks
                retrieved_chunks = await retrieval_func(current_query)
                # Assess retrieval
                assessment = await self.assess_retrieval(current_query, retrieved_chunks)
                process_log.append({
                    "step": "assessment",
                    "is_sufficient": assessment.is_sufficient,
                    "reasoning": assessment.reasoning
                })
                if assessment.is_sufficient:
                    response = await generation_func(current_query, retrieved_chunks)
                    return {
                        "success": True,
                        "response": response,
                        "process_log": process_log,
                        "final_query": current_query
                    }
                # Try decomposition if not sufficient
                sub_questions = await self.decompose_query(current_query)
                if len(sub_questions) > 1:
                    process_log.append({
                        "step": "decomposition",
                        "sub_questions": sub_questions
                    })
                    all_chunks = []
                    for sub_q in sub_questions:
                        sub_chunks = await retrieval_func(sub_q)
                        all_chunks.extend(sub_chunks)
                    # Assess sufficiency of combined chunks
                    assessment = await self.assess_retrieval(current_query, all_chunks)
                    process_log.append({
                        "step": "assessment_after_decomposition",
                        "is_sufficient": assessment.is_sufficient,
                        "reasoning": assessment.reasoning
                    })
                    if assessment.is_sufficient:
                        response = await generation_func(current_query, all_chunks)
                        return {
                            "success": True,
                            "response": response,
                            "process_log": process_log,
                            "final_query": current_query
                        }
                # If still not sufficient, try rewriting
                rewrite = await self.rewrite_query(
                    current_query,
                    f"Previous retrieval assessment: {assessment.reasoning}"
                )
                process_log.append({
                    "step": "query_rewrite",
                    "original": current_query,
                    "rewritten": rewrite.rewritten_query,
                    "reasoning": rewrite.reasoning
                })
                current_query = rewrite.rewritten_query
                retries += 1
            except Exception as e:
                self.logger.error(f"Error in process_query: {str(e)}")
                process_log.append({
                    "step": "error",
                    "error": str(e)
                })
                retries += 1
        return {
            "success": False,
            "error": "Max retries exceeded without finding sufficient context",
            "process_log": process_log,
            "final_query": current_query
        }