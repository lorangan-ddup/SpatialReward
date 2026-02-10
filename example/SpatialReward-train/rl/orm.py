import os
import re
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import json

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            content_to_parse = content_match.group(1).strip() if content_match else content
            has_answer_tag = content_match is not None

            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            sol_to_parse = sol_match.group(1).strip() if sol_match else sol

            gold_parsed = parse(sol_to_parse, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                if has_answer_tag:
                    answer_parsed = parse(content_to_parse, extraction_mode='first_match')
                else:
                    answer_parsed = parse(
                        content_to_parse,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    boxed=True,
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode='first_match',
                    )
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        response_token_ids = kwargs.get('response_token_ids')
        rewards = []
        for ids, acc_reward in zip(response_token_ids, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(ids)
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, soft_max_length, soft_cache_length):
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        response_token_ids = kwargs.get('response_token_ids')
        for ids in response_token_ids:
            completion_length = len(ids)
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards

class GeminiConsistencyReward(ORM):
    """
    Consistency/Aesthetic reward model using Gemini API.
    Calls gemini-3-flash-preview via OpenAI-compatible API.
    
    Returns only Rc reward, used to evaluate model reasoning consistency (hallucination check) or aesthetic quality.
    
    Evaluation mode is determined by tags in ground_truth:
    - Contains "sc": Consistency evaluation (requires original + edited images)
    - Contains "pq": Aesthetic evaluation (requires edited image only)
    """
    
    def __init__(self,
                 api_base_url: str = "http://path/to/gemini",
                 api_key: str = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
                 model_name: str = "gemini-3-flash-preview",
                 enable_jsonl_output: bool = False,
                 output_jsonl_path: Optional[str] = None):
        """
        Initialize Gemini Consistency Reward Model.
        
        Args:
            api_base_url: Base URL for OpenAI-compatible API
            api_key: API Key
            model_name: Model name
            enable_jsonl_output: Whether to enable JSONL output
            output_jsonl_path: Path for JSONL output
        """
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.model_name = model_name
        
        try:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=self.api_base_url,
                api_key=self.api_key
            )
        except Exception as e:
            print(f"warning: OpenAI client initialization failed: {e}")
            self.client = None
        
        self.enable_jsonl_output = enable_jsonl_output
        self.jsonl_writer = None
        if self.enable_jsonl_output:
            from swift.utils import JsonlWriter
            if output_jsonl_path is None:
                output_jsonl_path = "gemini_consistency_evaluations.jsonl"
            output_dir = os.path.dirname(os.path.abspath(output_jsonl_path))
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            self.jsonl_writer = JsonlWriter(output_jsonl_path)
            self.jsonl_writer.is_write_rank = True
            self.jsonl_writer.fpath = os.path.abspath(os.path.expanduser(output_jsonl_path))
            print(f"GeminiConsistencyReward JSONL output enabled: {output_jsonl_path}")
    
    @staticmethod
    def parse_output(text: str) -> dict:
        if '<think>' in text and '</think>' in text:
            think_end = text.index('</think>') + len('</think>')
            text = text[think_end:].strip()
        
        try:
            data = json.loads(text)
            return {
                'edit_region': data.get('edit_region', []),
                'reasoning': data.get('reasoning', ''),
                'score': data.get('score', [0, 0])
            }
        except json.JSONDecodeError:
            return {
                'edit_region': [],
                'reasoning': '',
                'score': [0, 0]
            }
    
    @staticmethod
    def get_mime_type(file_path: str) -> str:
        if file_path.lower().endswith('.png'):
            return 'image/png'
        elif file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
            return 'image/jpeg'
        elif file_path.lower().endswith('.webp'):
            return 'image/webp'
        return 'image/jpeg'
    
    @staticmethod
    def encode_image_to_base64(file_path: str) -> str:
        import base64
        with open(file_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def call_gemini(self, content_list: List) -> str:
        if self.client is None:
            return "ERROR: client_not_initialized"
        
        for retry in range(5):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": content_list}
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )
                
                if completion.choices[0].message.content:
                    return completion.choices[0].message.content
                
                return "ERROR: empty_response"
                
            except Exception as e:
                error_msg = str(e)
                
                if 'quota' in error_msg.lower() or 'limit' in error_msg.lower() or '429' in error_msg:
                    print(f"[WARNING] Quota limit, waiting and retrying ({retry + 1}/5)")
                    time.sleep(1)
                elif retry == 4:
                    print(f"[ERROR] Gemini API call failed: {e}")
                    return f"ERROR: {e}"
                else:
                    time.sleep(1)
        
        return "ERROR: max_retries_exceeded"
    
    def calculate_consistency_reward(self,
                                     reasoning: str,
                                     scores: List[float] = None,
                                     instruction: str = None,
                                     image_paths: List[str] = None,
                                     is_aesthetic_eval: bool = False) -> tuple:
        content_list = []
        if image_paths:
            for path_or_dict in image_paths:
                try:
                    if isinstance(path_or_dict, dict):
                        path = path_or_dict.get('path')
                        if path is None:
                            print(f"warning: dictionary does not have 'path' field: {path_or_dict}")
                            continue
                    else:
                        path = path_or_dict
                    
                    if os.path.exists(path):
                        base64_image = self.encode_image_to_base64(path)
                        mime_type = self.get_mime_type(path)
                        content_list.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{base64_image}"
                            }
                        })
                    else:
                        print(f"warning: image path does not exist: {path}")
                except Exception as e:
                    print(f"warning: failed to load image {path_or_dict}: {e}")
        
        # Build prompt
        if is_aesthetic_eval:
            prompt_text = f"""You are an expert evaluator for image editing tasks. Please assess whether the model's quality evaluation of the edited image is accurate.

Model Reasoning:
{reasoning}

Analyze the following aspects:
1. Factuality Check (Hallucination):
    - Do not label global blurriness as anatomy issues or AI artifacts.
    - Do not claim facial distortion if none exists.
    - Normal image blur or motion blur is NOT an artifact.
    - Do not hallucinate "grid/dot patterns" if they don't exist.
2. Logical Coherence: Is the reasoning consistent?
3. Objectivity: Is the quality assessment fair?

Return a brief evaluation reasoning and a quality score (0.0-1.0). If no issues, give 1; minor issues -0.5; severe issues 0. JSON Format:
{{
    "reasoning": "brief reasoning",
    "quality_score": 0.0-1.0
}}
"""
        else:
            scores_str = str(scores) if scores else "[0, 0]"
            prompt_text = f"""You are an expert evaluator for image editing tasks. Please assess whether the model's prediction on image editing is accurate and consistent.

The model provides reasoning on both "editing success" and "degree of over-editing". You need to focus on checking if the model hallucinates and if the score aligns with the reasoning.

Analyze the discrepancy between the model's reasoning and the actual image:
1. Consistency with Reality: Does the model fail to notice actual changes (e.g., changed facial expression) or describe changes incorrectly?
2. Over-editing Awareness: Does the model notice unintended changes not mentioned in the instruction (e.g., pose, composition, background)?
3. Fine-grained Details: Does the model overlook subtle inconsistencies in the edited subject?
4. Hallucination Check: Does the model mention non-existent subtle changes?
5. Logical Consistency: Is the reasoning self-contradictory?
6. False Over-editing: Does the model incorrectly flag necessary composition/pose changes (required by the instruction) as over-editing? (Over-editing score should not be too low for necessary changes).
7. Score-Reasoning Alignment: The scores must reflect the reasoning. (e.g., purely positive reasoning implies a success score > 20).

Editing Instruction: {instruction}

Model Reasoning:
{reasoning}

Model Scores [Success, Over-editing] (0-25): {scores_str}

Return a brief evaluation reasoning and a consistency score (0.0-1.0). If no issues, give 1; minor issues -0.5; severe issues 0. JSON Format:
{{ 
    "reasoning": "brief reasoning",
    "consistency_score": 0.0-1.0
}}
"""
        
        content_list.append({
            "type": "text",
            "text": prompt_text
        })
        
        result_text = self.call_gemini(content_list)
        
        if result_text.startswith("ERROR:"):
            print(f"warning: Gemini API call failed: {result_text}")
            gemini_eval = {
                'reasoning': f'API call failed: {result_text}',
                'score': 0.8,
                'eval_type': 'aesthetic' if is_aesthetic_eval else 'consistency',
                'raw_output': result_text
            }
            return 0.8, gemini_eval
        
        print(f"\n{'='*50}")
        print(f"Gemini original output:")
        print(result_text)
        print(f"{'='*50}\n")
        
        try:
            parse_text = result_text
            if '```json' in parse_text:
                parse_text = parse_text.split('```json')[1].split('```')[0].strip()
            elif '```' in parse_text:
                parse_text = parse_text.split('```')[1].split('```')[0].strip()
            
            result = json.loads(parse_text)
            
            if is_aesthetic_eval:
                Rc = result.get('quality_score', 0.5)
                gemini_reasoning = result.get('reasoning', '')
                print(f"Aesthetic Eval - Gemini Reasoning: {gemini_reasoning}")
                print(f"Aesthetic Eval - Quality Score: {Rc}")
            else:
                Rc = result.get('consistency_score', 0.5)
                gemini_reasoning = result.get('reasoning', '')
                print(f"Consistency Eval - Gemini Reasoning: {gemini_reasoning}")
                print(f"Consistency Eval - Consistency Score: {Rc}")
            
            Rc = max(0.0, min(1.0, Rc))
            
            gemini_eval = {
                'reasoning': gemini_reasoning,
                'score': Rc,
                'eval_type': 'aesthetic' if is_aesthetic_eval else 'consistency',
                'raw_output': result_text
            }
            
            return Rc, gemini_eval
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            gemini_eval = {
                'reasoning': f'JSON parsing failed: {e}',
                'score': 0.5,
                'eval_type': 'aesthetic' if is_aesthetic_eval else 'consistency',
                'raw_output': result_text
            }
            return 0.5, gemini_eval
    
    def __call__(self,
                 infer_requests: List[Union['InferRequest', Dict]],
                 solution: List[str],
                 **kwargs) -> List[float]:
        """
        Calculate Gemini Consistency/Aesthetic Reward.
        
        Returns only Rc reward.
        
        Evaluated mode detection:
        - If ground_truth contains "sc": Consistency evaluation
        - If ground_truth contains "pq": Aesthetic evaluation
        
        Args:
            infer_requests: List of inference requests or dicts
            solution: List of solution strings, containing only type tags ("sc" or "pq")
            
        Returns:
            List of rewards (only Rc)
        """
        rewards = []
        
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        
        for i, (prediction, ground_truth) in enumerate(zip(predictions, solution)):
            try:
                pred_data = self.parse_output(prediction)
                
                pred_reasoning = pred_data['reasoning']
                pred_score = pred_data['score']
                
                instruction_list = kwargs.get('instruction')
                images_list = kwargs.get('images')
                id_list = kwargs.get('id')
                
                if instruction_list is None:
                    raise ValueError("must pass 'instruction' parameter through kwargs")
                if images_list is None:
                    raise ValueError("must pass 'images' parameter through kwargs")
                
                instruction = instruction_list[i]
                image_paths = images_list[i]
                sample_id = id_list[i] if id_list and i < len(id_list) else None
                
                is_aesthetic_eval = 'pq' in ground_truth.lower()
                
                Rc, gemini_eval = self.calculate_consistency_reward(
                    pred_reasoning,
                    pred_score,
                    instruction,
                    image_paths,
                    is_aesthetic_eval
                )
                
                if self.enable_jsonl_output and self.jsonl_writer is not None:
                    try:
                        record = {
                            'id': sample_id,  
                            'sample_index': i,
                            'instruction': instruction,
                            'image_paths': [
                                str(p) if not isinstance(p, dict) else p.get('path', '')
                                for p in image_paths
                            ],
                            'model_output': {
                                'reasoning': pred_reasoning,
                                'score': pred_score,
                                'edit_region': pred_data['edit_region']
                            },
                            'ground_truth_type': ground_truth,
                            'gemini_evaluation': gemini_eval,
                            'rewards': {
                                'Rc': float(Rc)
                            },
                            'eval_mode': 'aesthetic' if is_aesthetic_eval else 'consistency'
                        }
                        self.jsonl_writer.append(record)
                
                # Print reward info
                print(f"\n{'='*60}")
                print(f"Sample {i+1} - {'Aesthetic Eval' if is_aesthetic_eval else 'Consistency Eval'}")
                print(f"  Consistency/Aesthetic Reward - Rc: {Rc:.4f}")
                print(f"{'='*60}\n")
                
                rewards.append(float(Rc))
                
            except Exception as e:
                print(f"failed to calculate reward: {e}")
                rewards.append(0.0)
        
        return rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'gemini_consistency': GeminiConsistencyReward,
}
