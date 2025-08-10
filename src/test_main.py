import os
from dotenv import load_dotenv

import deepeval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval


load_dotenv()
deepeval.login(os.getenv("DEEP_API_KEY"))



test_case = LLMTestCase(input="1. Address", 
                        actual_output="A/P-123 Main St, Springfield, USA", 
                        expected_output="A/P-123 Main St, Springfield, USA Mob:917 589 2998",)
address_metric = GEval(
    name="Address Identification",
    criteria="Identify the address.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT]
)
address_metric.measure(test_case)
print(address_metric.score, address_metric.reason)