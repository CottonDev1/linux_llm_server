"""
Content Analysis Service

Analyzes call transcription using LLM to extract subject, outcome, and customer name.
"""

import os
import json
from typing import Dict, Optional

# HTTP client for LLM API
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    print("Warning: httpx not installed. Content analysis will be disabled.")


class ContentAnalysisService:
    """
    Service for analyzing call content using LLM.

    Extracts:
    - Subject: What the call was about
    - Outcome: How the call ended (resolved, unresolved, etc.)
    - Customer Name: Name of the customer (not staff)
    """

    _instance: Optional['ContentAnalysisService'] = None

    def __init__(self):
        self.llm_url = os.getenv("LLAMACPP_HOST", "http://localhost:8081")

    @classmethod
    def get_instance(cls) -> 'ContentAnalysisService':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def analyze_call_content(
        self,
        transcription: str,
        duration_seconds: float,
        staff_name: str = None
    ) -> Dict:
        """
        Analyze call transcription using LLM to extract subject, outcome, and customer name.

        Args:
            transcription: The call transcription text
            duration_seconds: Duration of the audio
            staff_name: Optional staff member name - if customer_name matches this, it will be cleared

        Returns:
            Dict with subject, outcome, customer_name, confidence, and analysis_model
        """
        result = {
            "subject": None,
            "outcome": None,
            "customer_name": None,
            "confidence": 0.0,
            "analysis_model": ""
        }

        if not HTTPX_AVAILABLE:
            print("httpx not available, skipping call content analysis")
            return result

        if not transcription or len(transcription.strip()) < 50:
            print("Transcription too short for content analysis")
            return result

        # Truncate very long transcriptions
        text_to_analyze = transcription[:6000] if len(transcription) > 6000 else transcription

        prompt = f"""Analyze this customer support call transcription and extract the following information.
Respond ONLY with a JSON object in exactly this format, no other text:

{{
  "subject": "brief description of what the call was about (1-2 sentences)",
  "outcome": "one of: Resolved, Unresolved, Pending Follow-up, Information Provided, Transferred, or Unknown",
  "customer_name": "the CUSTOMER's name (the person calling for help), NOT the support staff name, otherwise null"
}}

IMPORTANT: The customer_name should be the person CALLING for support, not the support representative answering the call.
If someone says "Hi this is [Name] from EWR" or "This is [Name] with support", that is the STAFF, not the customer.
The customer is typically the one asking questions or requesting help.

Call Duration: {duration_seconds:.0f} seconds

Transcription:
{text_to_analyze}

JSON Response:"""

        try:
            timeout_config = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)

            async with httpx.AsyncClient(timeout=timeout_config) as client:
                response = await client.post(
                    f"{self.llm_url}/v1/completions",
                    json={
                        "prompt": prompt,
                        "max_tokens": 300,
                        "temperature": 0.1,
                        "stop": ["\n\n", "```"]
                    }
                )
                response.raise_for_status()
                data = response.json()
                content = data.get('choices', [{}])[0].get('text', '').strip()

                if content:
                    try:
                        # Clean up the response - remove any markdown formatting
                        json_str = content
                        if json_str.startswith("```"):
                            json_str = json_str.split("```")[1]
                            if json_str.startswith("json"):
                                json_str = json_str[4:]
                        json_str = json_str.strip()

                        # Find JSON object boundaries
                        start_idx = json_str.find('{')
                        end_idx = json_str.rfind('}')
                        if start_idx != -1 and end_idx != -1:
                            json_str = json_str[start_idx:end_idx + 1]

                        parsed = json.loads(json_str)

                        result["subject"] = parsed.get('subject', '').strip() if parsed.get('subject') else None
                        result["outcome"] = parsed.get('outcome', '').strip() if parsed.get('outcome') else None
                        result["customer_name"] = parsed.get('customer_name', '').strip() if parsed.get('customer_name') else None
                        result["confidence"] = 0.8
                        result["analysis_model"] = "Llama-3.2-3B-Instruct"

                        # Validate customer_name is not the staff member's name
                        if result["customer_name"] and staff_name:
                            if self._is_same_person(result["customer_name"], staff_name):
                                print(f"Customer name '{result['customer_name']}' matches staff name '{staff_name}', clearing")
                                result["customer_name"] = None

                        subject_preview = result['subject'][:50] if result['subject'] else 'N/A'
                        print(f"Call content analysis complete: subject='{subject_preview}...', outcome='{result['outcome']}', customer='{result['customer_name'] or 'N/A'}'")

                    except json.JSONDecodeError as e:
                        print(f"Failed to parse LLM response as JSON: {e}")
                        result["subject"] = content[:200] if content else None
                        result["confidence"] = 0.3
                        result["analysis_model"] = "Llama-3.2-3B-Instruct"

        except httpx.TimeoutException:
            print("Call content analysis timed out")
        except Exception as e:
            print(f"Call content analysis failed: {e}")

        return result

    def _is_same_person(self, name1: str, name2: str) -> bool:
        """
        Check if two names likely refer to the same person.

        Args:
            name1: First name
            name2: Second name

        Returns:
            True if names likely refer to same person
        """
        if not name1 or not name2:
            return False

        n1_lower = name1.lower().strip()
        n2_lower = name2.lower().strip()

        # Direct match
        if n1_lower == n2_lower:
            return True

        # First name match
        n1_parts = n1_lower.split()
        n2_parts = n2_lower.split()

        if n1_parts and n2_parts and n1_parts[0] == n2_parts[0]:
            return True

        # Partial containment
        if n1_lower in n2_lower or n2_lower in n1_lower:
            return True

        return False


def get_content_analysis_service() -> ContentAnalysisService:
    """Get the singleton content analysis service instance"""
    return ContentAnalysisService.get_instance()
