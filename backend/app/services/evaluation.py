"""
Evaluation Service - RAG Quality Metrics

Computes quality metrics for RAG responses:
- Faithfulness: Is the answer grounded in the source documents?
- Relevance: Are the retrieved sources relevant to the query?
- Answer Quality: Overall response quality assessment
"""
from typing import Dict, Any, List
from app.services.github_models import github_models
import json
import logging

logger = logging.getLogger(__name__)


class EvaluationService:
    """Evaluates RAG pipeline quality using LLM-as-judge."""
    
    FAITHFULNESS_PROMPT = """You are an evaluation judge. Given a question, an answer, and source documents, 
rate how faithful the answer is to the sources.

Faithfulness means: Every claim in the answer can be traced back to the source documents.

Question: {question}

Answer: {answer}

Sources:
{sources}

Rate faithfulness from 0.0 to 1.0 where:
- 1.0 = Every claim is directly supported by sources
- 0.5 = Some claims are supported, some are not
- 0.0 = Answer contradicts or has no basis in sources

Respond ONLY with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

    RELEVANCE_PROMPT = """You are an evaluation judge. Given a question and retrieved source documents,
rate how relevant the sources are to answering the question.

Question: {question}

Sources:
{sources}

Rate relevance from 0.0 to 1.0 where:
- 1.0 = Sources directly address the question with perfect information
- 0.5 = Sources are somewhat related but don't fully address the question
- 0.0 = Sources are completely irrelevant

Respond ONLY with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}"""

    QUALITY_PROMPT = """You are an evaluation judge. Rate the overall quality of this AI response.

Question: {question}

Answer: {answer}

Rate quality from 0.0 to 1.0 considering:
- Accuracy and correctness
- Completeness of the answer
- Clarity and organization
- Helpfulness

Respond ONLY with JSON: {{"score": <float>, "reasoning": "<brief explanation>"}}"""
    
    def evaluate(
        self, 
        question: str, 
        answer: str, 
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite on a RAG response.
        
        Args:
            question: The user's original question
            answer: The AI-generated answer
            sources: List of retrieved source documents
            
        Returns:
            Dict with faithfulness, relevance, and quality scores
        """
        sources_text = "\n\n".join([
            f"[Source {i+1}]: {s.get('content', '')}" 
            for i, s in enumerate(sources)
        ]) if sources else "[No sources used]"
        
        # Run all evaluations
        faithfulness = self._evaluate_metric(
            self.FAITHFULNESS_PROMPT.format(
                question=question, answer=answer, sources=sources_text
            )
        )
        
        relevance = self._evaluate_metric(
            self.RELEVANCE_PROMPT.format(
                question=question, sources=sources_text
            )
        )
        
        quality = self._evaluate_metric(
            self.QUALITY_PROMPT.format(
                question=question, answer=answer
            )
        )
        
        return {
            "faithfulness": faithfulness,
            "relevance": relevance,
            "answer_quality": quality,
            "overall_score": round(
                (faithfulness["score"] + relevance["score"] + quality["score"]) / 3, 3
            )
        }
    
    def _evaluate_metric(self, prompt: str) -> Dict[str, Any]:
        """Evaluate a single metric using LLM-as-judge."""
        try:
            messages = [
                {"role": "system", "content": "You are a precise evaluation judge. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ]
            response = github_models.chat_completion(messages, temperature=0.0, max_tokens=200)
            result = json.loads(response.strip())
            return {
                "score": round(float(result.get("score", 0)), 3),
                "reasoning": result.get("reasoning", "No reasoning provided")
            }
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Evaluation failed: {e}")
            return {"score": 0.0, "reasoning": f"Evaluation error: {str(e)}"}


# Singleton instance
evaluation_service = EvaluationService()
