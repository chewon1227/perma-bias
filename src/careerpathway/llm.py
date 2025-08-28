from careerpathway.utils import load_api_config, set_api_keys
import os
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from careerpathway.utils import open_json, save_json
from tqdm import tqdm, trange
import anthropic
from openai import OpenAI
from vllm import LLM, SamplingParams
import google.generativeai as genai
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Union

load_api_config()
set_api_keys()


MAX_TOKENS = 100
TEMPERATURE = 0.1
TOP_P = 1.0
TOP_K = 0


MODELS = {
    "gpt": ["gpt-4-turbo", "gpt-4o-mini", 'gpt-4o', 'o1', 'o1-mini'],
    "gemini": ["gemini-1.0-pro"],
    "anthropic": ["claude-3-5-sonnet-20240620", "claude-3-5-sonnet-20241022"],
    "hf": [],  # can't vllm
}


def get_results(
    model_name_or_path: str,
    prompt: str,
    data: list[dict],
    dtype: str = 'auto',
    max_tokens: int = 100,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    save_path: str = None,
    batch_size: int = 1,
    do_log: bool = False,
    apply_chat_template : bool | str = 'auto',
    system_prompt : str = None,
    while_loop: bool = False,
    max_num_seqs: int = 2000,
    max_model_len: int = 1024,
    gpu_memory_utilization: float = 0.95,
    max_workers: int = 20,  # Number of parallel workers for API models
):

    L = MyLLMEval(model_name_or_path, dtype=dtype, max_num_seqs=max_num_seqs, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization)
    results = L.inference(
        prompt=prompt,
        data=data,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        save_path=save_path,
        batch_size=batch_size,
        do_log=do_log,
        while_loop=while_loop,
        apply_chat_template=L.apply_chat_template if apply_chat_template=='auto' else apply_chat_template,
        system_prompt=system_prompt,
        max_workers=max_workers,  # Pass max_workers to inference
    )
    return results


class MyLLMEval:
    def __init__(self, model_path : str, dtype: str = 'auto', task: str = None, 
                 gpu_memory_utilization: float = 0.90, max_num_seqs: int = 50, max_model_len: int = 8192):
        self.model_path = model_path
        self.type = self.gettype()
        self.max_tokens = MAX_TOKENS
        self.temperature = TEMPERATURE
        self.while_loop = False
        self.top_p = TOP_P
        self.top_k = TOP_K
        self.apply_chat_template = 'auto'
        self.dtype = dtype
        self.task = task

        print(f"Model path: {self.model_path} type: {self.type}")
        
        if self.type == "gemini":
            self.model = None
            self.tokenizer = None
            self.apply_chat_template = False
            
        elif self.type == 'anthropic':
            self.model = None
            self.tokenizer = None
            self.apply_chat_template = False
            
        elif self.type == "gpt":
            self.client = OpenAI()
            self.tokenizer = None
            self.apply_chat_template = False
            
        elif self.type == "hf":
            self.apply_chat_template = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path
            )  # the tokenizer should match that of the index you load below
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
                token=os.getenv("HUGGINGFACE_API_KEY"),
            )  # please replace index_dir with the local directory where you store the index
            
        else:
            if self.task == "embed":
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=torch.cuda.device_count(),
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=self.dtype,
                    task="embed"
                )
            else:
                self.llm = LLM(
                    model=self.model_path,
                    tensor_parallel_size=torch.cuda.device_count(),
                    max_num_seqs=max_num_seqs,
                    max_model_len=max_model_len,
                    gpu_memory_utilization=gpu_memory_utilization,
                    dtype=self.dtype,
                )
                print(f"VLLM max_num_seqs: {max_num_seqs}, max_model_len: {max_model_len}, gpu_memory_utilization: {gpu_memory_utilization}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.apply_chat_template = True


    def gettype(self):
        for key, value in MODELS.items():
            if self.model_path in value:
                return key
        else:
            return "vllm"


    def chat_template(self, data: list[dict] | dict, prompt: str, system_prompt: str) -> list[str]:
        if isinstance(data, dict):
            data = [data]
        if system_prompt is None:
            messages = [
                [{'role' : 'user', 'content' : prompt.format(*item['inputs'])}]
                for item in data
            ]
        else:
            messages = [
                [
                    {'role' : 'system', 'content' : system_prompt},
                    {'role' : 'user', 'content' : prompt.format(*item['inputs'])}
                    ]
                for item in data
            ]
        return [self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False) for message in messages]
        
        
    def generate(self, input_text: str | list[str], system_prompt: str = None) -> str | list[str]:
        self.GENERATEFUNC = {
            "gpt": self.gptgenerate,
            'anthropic': self.anthropicgenerate,
            "gemini": self.geminigenerate,
            "hf": self.Hfgenerate,
            "vllm": self.vllmgenerate,
        }
        generatefunc = self.GENERATEFUNC[self.type]
        if self.while_loop:
            while True:
                try:
                    res = generatefunc(input_text, system_prompt)
                    return res
                except Exception as e:
                    print("While trying with error", e)
                    time.sleep(5)
                    continue
        else:       
            return generatefunc(input_text, system_prompt)


    def geminigenerate(self, input_text: str, system_prompt: str | None) -> str:
        safety_settings = [
            {
                "category": "HARM_CATEGORY_DANGEROUS",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_NONE",
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE",
            },
        ]
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name=self.model_path)
        if system_prompt != None:
            input_text = f"{system_prompt}\n{input_text}"
        response = model.generate_content(input_text)
        time.sleep(1)
        return response.text

    
    def anthropicgenerate(self, input_text: str, system_prompt: str | None) -> str:
        client = anthropic.Anthropic(
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        )
        if system_prompt != None:
            messages = [{'role' : 'system', 'content' : system_prompt}, {'role' : 'user', 'content' : input_text}]
        else:
            messages = [{'role' : 'user', 'content' : input_text}]
        message = client.messages.create(
            model=self.model_path,
            max_tokens=self.max_tokens,
            messages=messages
        )
        return message.content[0].text


    def gptgenerate(self, input_text: str, system_prompt: str | None ) -> str:
        if system_prompt != None:
            messages = [{'role' : 'system', 'content' : system_prompt}, {'role' : 'user', 'content' : input_text}]
        else:
            messages = [{'role' : 'user', 'content' : input_text}]
        if self.model_path == 'o1':
            kwargs = {
                'top_p' : self.top_p,
            }
        else:
            kwargs = {
                'temperature' : self.temperature,
                'max_completion_tokens' : self.max_tokens,
                'top_p' : self.top_p,
            }
        completion = self.client.chat.completions.create(
            model=self.model_path,
            messages=messages,
            **kwargs
        )
        return completion.choices[0].message.content


    def Hfgenerate(self, input_text: str, system_prompt : str | None) -> str:
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
        return self.tokenizer.decode(
            output[0][-1 * len(input_ids) :], skip_special_tokens=True
        )


    def vllmgenerate(self, input_text: str | list, system_prompt: str | None) -> str | list[str]:
        sampling_params = SamplingParams(
            temperature=self.temperature, max_tokens=self.max_tokens, seed=self.random_seed if self.random_seed is not None else 0,
        )
        if isinstance(input_text, str):
            outputs = self.llm.generate([input_text], sampling_params)
            return outputs[0].outputs[0].text
        else:
            outputs = self.llm.generate(input_text, sampling_params)
            return [output.outputs[0].text for output in outputs]


    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Get embeddings for a list of texts using vLLM."""
        if not isinstance(texts, list):
            texts = [texts]
            
        # For embedding models, we use a different approach
        if hasattr(self.llm, 'embed'):
            outputs = self.llm.embed(texts)
            embeddings = [output.outputs.embedding for output in outputs]
            return np.array(embeddings)
            
        # Fallback to using hidden states if embed is not available
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=1,  # We only need the encoder output
        )
        
        outputs = self.llm.generate(texts, sampling_params)
        
        # Get the last hidden states
        hidden_states = []
        for output in outputs:
            hidden_state = output.outputs[0].hidden_states[-1]  # Get last layer hidden state
            # Mean pooling over sequence length
            pooled = hidden_state.mean(dim=0)
            hidden_states.append(pooled)
            
        # Stack all hidden states
        embeddings = torch.stack(hidden_states)
        return embeddings.cpu().numpy()

    def embed(self, texts: list[str]) -> list[dict]:
        """Wrapper method to match the example code's usage pattern."""
        embeddings = self.get_embeddings(texts)
        return [{"outputs": {"embedding": emb}} for emb in embeddings]

    def compute_similarity(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and candidates."""
        # Normalize the embeddings
        query_norm = np.linalg.norm(query_embedding)
        candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
        
        # Compute cosine similarity
        similarities = np.dot(candidate_embeddings, query_embedding) / (query_norm * candidate_norms)
        return similarities

    def _process_single_item(self, input_item, prompt, system_prompt, index):
        """Process a single input item and return the result with index for ordering."""
        try:
            if self.apply_chat_template:
                input_text = self.chat_template(input_item, prompt, system_prompt)
            else:
                input_text = prompt.format(*input_item['inputs'])
            if isinstance(input_text, list) and len(input_text) == 1:
                input_text = input_text[0]
            output = self.generate(input_text, system_prompt=system_prompt)
            result = {
                'prompt' : input_text,
                'result' : output,
                **{k : v for k, v in input_item.items() if k != 'inputs'},
                'hyperparameters' : {
                    'model' : self.model_path,
                    'max_tokens' : self.max_tokens,
                    'temperature' : self.temperature,
                    'top_p' : self.top_p,
                    'top_k' : self.top_k,
                    'apply_chat_template' : self.apply_chat_template,
                    'system_prompt' : system_prompt
                },
                'index': index  # For maintaining order
            }
            return result
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            return None

    def _single_inference(self, prompt: str, data: list[dict], system_prompt: str) -> list[dict]:
        results = []
        
        # Use parallel processing for API models
        if self.type in ["gpt", "gemini", "anthropic"] and self.max_workers > 1:
            return self._parallel_inference(prompt, data, system_prompt, self.max_workers)
        
        # Sequential processing for other models
        for input_item in tqdm(data):
            if self.apply_chat_template:
                input_text = self.chat_template(input_item, prompt, system_prompt)
            else:
                input_text = prompt.format(*input_item['inputs'])
            if isinstance(input_text, list) and len(input_text) == 1:
                input_text = input_text[0]
            output = self.generate(input_text, system_prompt=system_prompt)
            output = {
                'prompt' : input_text,
                'result' : output,
                **{k : v for k, v in input_item.items() if k != 'inputs'},
                'hyperparameters' : {
                    'model' : self.model_path,
                    'max_tokens' : self.max_tokens,
                    'temperature' : self.temperature,
                    'top_p' : self.top_p,
                    'top_k' : self.top_k,
                    'apply_chat_template' : self.apply_chat_template,
                    'system_prompt' : system_prompt
                }
            }
            if self.save_additionally:
                save_json([output], self.save_path, mode='a')
            else:
                results.append(output)
                save_json(results, self.save_path, mode='w')
                
        return [r['result'] for r in results] if not self.save_additionally else None

    def _parallel_inference(self, prompt: str, data: list[dict], system_prompt: str, max_workers: int = 5) -> list[dict]:
        """Parallel inference for API-based models using ThreadPoolExecutor."""
        results = [None] * len(data)  # Pre-allocate to maintain order
        completed_results = []
        file_lock = Lock()  # For thread-safe file operations
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._process_single_item, input_item, prompt, system_prompt, i): i 
                for i, input_item in enumerate(data)
            }
            
            # Process completed tasks
            for future in tqdm(as_completed(future_to_index), total=len(data), desc="Processing"):
                index = future_to_index[future]
                try:
                    result = future.result()
                    if result is not None:
                        results[index] = result
                        
                        # Thread-safe saving
                        with file_lock:
                            if self.save_additionally:
                                # Remove index before saving
                                save_result = {k: v for k, v in result.items() if k != 'index'}
                                save_json([save_result], self.save_path, mode='a')
                            else:
                                # Collect completed results for batch saving
                                completed_results.append(result)
                                # Save all completed results so far (maintaining order)
                                ordered_results = [r for r in results if r is not None]
                                ordered_results = [{k: v for k, v in r.items() if k != 'index'} 
                                                 for r in ordered_results]
                                save_json(ordered_results, self.save_path, mode='w')
                        
                except Exception as e:
                    print(f"Error processing future for index {index}: {e}")
        
        # Filter out None results and remove index field
        final_results = [r for r in results if r is not None]
        final_results = [{k: v for k, v in r.items() if k != 'index'} for r in final_results]
        
        return [r['result'] for r in final_results] if not self.save_additionally else None

    def _batch_inference(self, prompt: str, data: list[dict], batch_size: int, system_prompt: str) -> list[dict]:
        results = []
        for i in trange(0, len(data), batch_size):
            inputs = data[i : i + batch_size]
            if self.apply_chat_template:
                chat_templated_inputs = self.chat_template(inputs, prompt, system_prompt)
            else:
                chat_templated_inputs = [prompt.format(*input['inputs']) for input in inputs]
            outputs = self.generate(chat_templated_inputs, system_prompt)
            outputs = [{
                'prompt' : prompt.format(*input['inputs']),
                'result' : output,
                **{k : v for k, v in input.items() if k != 'inputs'},
                'hyperparameters' : {
                    'model' : self.model_path,
                    'max_tokens' : self.max_tokens,
                    'temperature' : self.temperature,
                    'top_p' : self.top_p,
                    'top_k' : self.top_k,
                    'apply_chat_template' : self.apply_chat_template,
                    'system_prompt' : system_prompt
            }
            }
                for input, output in zip(inputs, outputs)
            ]
            if self.save_additionally:
                save_json(outputs, self.save_path, mode='a')
            else:
                results.extend(outputs)
                save_json(results, self.save_path, mode='w')
            
            if self.do_log:
                if i < 30:
                    print(f"Inputs: {chat_templated_inputs[0]}\nOutput: {outputs[0]}")
        return [r['result'] for r in results] if not self.save_additionally else None


    def inference(
        self,
        prompt: str,
        data: list[dict],
        max_tokens: int = 100,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        save_path: str = None,
        batch_size: int = 1,
        do_log: bool = False,
        while_loop: bool = False,
        apply_chat_template : bool = True,
        system_prompt : str = None,
        save_additionally: bool = True,
        random_seed: int = None,
        max_workers: int = 20,  # Number of parallel workers for API models
    ):
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.while_loop = while_loop
        self.top_p = top_p
        self.top_k = top_k
        self.random_seed = random_seed
        self.apply_chat_template = apply_chat_template
        self.max_workers = max_workers  # Store for use in parallel inference
        if save_path is None:
            save_path = os.path.join("results", f"{time.time()}.json")
        self.save_path = save_path
        self.save_additionally = save_additionally
        self.do_log = do_log
        if (self.type == "vllm") and (batch_size > 1):
            return self._batch_inference(prompt, data, batch_size, system_prompt)
        else:
            return self._single_inference(prompt, data, system_prompt)
