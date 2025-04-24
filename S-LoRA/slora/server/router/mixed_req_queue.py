import uuid
import numpy as np
from typing import List, Optional

# Example import if you have a local definition:
# from ..io_struct import Batch, Req
from ..io_struct import Batch, Req
from ..tokenizer import get_tokenizer
from ..input_params import FinetuneParams
from ..sampling_params import SamplingParams

# If using the original time calculation decorator
# from slora.utils.infer_utils import calculate_time

def get_finetuning_sampling_params() -> SamplingParams:
    """
    Return a 'dummy' sampling params object suitable for fine-tuning requests.
    By default, no sampling or advanced penalties are used.
    """
    sp = SamplingParams(
        do_sample=False,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        ignore_eos=False,     # Typically handle EOS in training
        max_new_tokens=1,   # Arbitrary placeholder
        stop_sequences=[]
    )
    return sp


def rprint(*args):
    return
    RED = "\033[91m"
    RESET = "\033[0m"
    print(RED + " ".join(str(arg) for arg in args) + RESET)

class Mixed_ReqQueue:
    """
    A queue that handles both inference requests and fine-tuning requests.
    Key differences from ReqQueue:
    - We store inference requests in `waiting_req_list`.
    - We store fine-tuning requests in `finetuning_req_list`.
    - We only add fine-tuning requests if (and only if) there are no inference
      requests waiting.
    """

    def __init__(self,
                 max_total_tokens: int,
                 batch_max_tokens: int,
                 running_max_req_size: int,
                 finetune_params: FinetuneParams):
        self.max_total_tokens = max_total_tokens
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.finetuning_data_path = finetune_params.finetuning_data_path
        self.finetuning_prepare_size = finetune_params.finetuning_prepare_size
        self.finetuning_lora_path = finetune_params.finetuning_lora_path  
        self.finetuning_pool_size = 8
        self.repeat_file = 100
    
        try: 
            self.tokenizer = get_tokenizer(finetune_params.model_weightdir, 
                                           finetune_params.tokenizor_mode, 
                                           trust_remote_code=finetune_params.trust_remote_code) 
        except:
            print("Could not load tokenizer. Using default.")
            self.tokenizer = get_tokenizer("huggyllama/llama-7b", finetune_params.tokenizor_mode) 

        # Holds waiting inference requests
        self.waiting_req_list: List[Req] = []
        # Holds waiting fine-tuning requests
        self.finetuning_req_list: List[Req] = []
        
        # Used internally to track concurrency usage when building a batch
        self.cache_len_list = []
        self.adapters = set()
        self.adapter_size = 0
        self.prepare_finetuning_requests()

    def append(self, req: Req):
        self.waiting_req_list.append(req)

    def prepare_finetuning_requests(self):
        loaded_count = 0
        try:
            finetune_req_buffer = []
            with open(self.finetuning_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    text = line.strip()
                    if not text:
                        continue
                    
                    # Tokenize the text
                    tokens = self.tokenizer(text)
                    prompt_ids = tokens.get('input_ids', [])
                    
                    new_req = Req(
                        adapter_dir=self.finetuning_lora_path,                
                        request_id=uuid.uuid4().hex,     # Unique request ID
                        prompt_ids=prompt_ids,
                        sample_params=get_finetuning_sampling_params(),
                        is_finetuning=True,               # Mark this request as fine-tuning
                        text=text,  # Store the original text for reference
                    )
                    
                    # Append to our fine-tuning queue
                    finetune_req_buffer.append(new_req)

                    loaded_count += 1
                    if loaded_count >= self.finetuning_prepare_size:
                        break
                
                for i in range(self.repeat_file):
                    for req in finetune_req_buffer:
                        new_req = Req(
                            adapter_dir=self.finetuning_lora_path,                
                            request_id=uuid.uuid4().hex,     # Unique request ID
                            prompt_ids=req.prompt_ids[:],
                            sample_params=get_finetuning_sampling_params(),
                            is_finetuning=True,               # Mark this request as fine-tuning
                            text=req.text,
                        )
                        self.finetuning_req_list.append(new_req)
                        loaded_count += 1
                print(f"Loaded {loaded_count} fine-tuning requests.")
        except FileNotFoundError:
            print(f"Could not find finetuning_data_path: {self.finetuning_data_path}")
        except Exception as e:
            print(f"Error reading or tokenizing finetuning data: {e}")

    def _init_cache_list(self, current_batch: Optional[Batch], lora_ranks: dict[str, int]):
        if current_batch is not None:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0
            for req in current_batch.reqs:
                used_len = req.input_len + len(req.output_ids)
                left_len = req.max_output_len - len(req.output_ids) - 1
                self.cache_len_list.append((used_len, left_len))

                if req.adapter_dir not in self.adapters and req.adapter_dir is not None:
                    self.adapter_size += lora_ranks[req.adapter_dir] * 4
                    self.adapters.add(req.adapter_dir)
        else:
            self.cache_len_list = []
            self.adapters = set()
            self.adapter_size = 0

    def _can_add_new_req(self, req: Req, lora_ranks: dict[str, int]) -> bool:
        self.cache_len_list.append((req.input_len + 1, req.max_output_len - 1))
        self.cache_len_list.sort(key=lambda x: -x[1])
    
        if req.adapter_dir not in self.adapters and req.adapter_dir is not None:
            self.adapter_size += lora_ranks[req.adapter_dir] * 4
            self.adapters.add(req.adapter_dir)
        
        left_out_len_array = np.array([e[1] for e in self.cache_len_list])
        has_run_len_array  = np.array([e[0] for e in self.cache_len_list])
        cum_run_len_array  = np.cumsum(has_run_len_array)
        size_array         = np.arange(1, len(self.cache_len_list) + 1)

        need_max_token_num = (left_out_len_array * size_array + cum_run_len_array).max()

        if (need_max_token_num < (self.max_total_tokens - self.adapter_size) and
            len(self.cache_len_list) <= self.running_max_req_size):
            return True
        else:
            return False

    def update_counter(self, req: Req):
        pass

    def generate_new_batch(self, current_batch: Optional[Batch], lora_ranks: dict[str, int]) -> Optional[Batch]:
        """
        Generates a new batch. Priority:
          1) If inference requests exist in `self.waiting_req_list`, only fill them.
          2) Otherwise, fill with fine-tuning requests from `self.finetuning_req_list`.
        """
        if current_batch is not None and len(current_batch.reqs) >= self.running_max_req_size:
            return None

        self._init_cache_list(current_batch, lora_ranks)
        
        # 1) If we have inference requests, try them first
        if len(self.waiting_req_list) > 0:
            can_run_list = []
            new_batch_total_tokens = 0
            aborted_count = 0
            
            for req in self.waiting_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            
            if len(can_run_list) > 0:
                new_batch = Batch(uuid.uuid4().hex, can_run_list)
                # Remove used + aborted from the front
                self.waiting_req_list = self.waiting_req_list[len(can_run_list) + aborted_count:]
                return new_batch
            else:
                # We found inference requests but couldn't fit any, so do nothing
                return None

        else:
            # 2) Otherwise, try fine-tuning requests
            can_run_list = []
            new_batch_total_tokens = 0
            aborted_count = 0
            if(len(self.finetuning_req_list)>0):
                rprint("Adding finetuning requests, #available requests in list", len(self.finetuning_req_list))
            for index, req in enumerate(self.finetuning_req_list):
                if len(can_run_list) >= self.finetuning_pool_size:
                    break
                if req.aborted:
                    aborted_count += 1
                    continue
                if ((new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break

            if len(can_run_list) > 0:
                can_run_list[-1].is_finetuning = False
                can_run_list[-1].needs_to_notify_detokenize = True
                print("Inference request text:", can_run_list[-1].text)
                new_batch = Batch(uuid.uuid4().hex, can_run_list)
                self.finetuning_req_list = self.finetuning_req_list[len(can_run_list) + aborted_count:]
                return new_batch
            else:
                return None

    def next_batch(self) -> Optional[Batch]:
        if len(self.waiting_req_list) > 0:
            next_batch_reqs = []
            new_batch_total_tokens = 0
            aborted_count = 0

            for req in self.waiting_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                    next_batch_reqs.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break

            if len(next_batch_reqs) > 0:
                new_batch = Batch(uuid.uuid4().hex, next_batch_reqs)
                self.waiting_req_list = self.waiting_req_list[len(next_batch_reqs) + aborted_count:]
                return new_batch
            else:
                return None
        else:
            next_batch_reqs = []
            new_batch_total_tokens = 0
            aborted_count = 0

            for req in self.finetuning_req_list:
                if req.aborted:
                    aborted_count += 1
                    continue
                if new_batch_total_tokens + req.input_len <= self.batch_max_tokens:
                    next_batch_reqs.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break

            if len(next_batch_reqs) > 0:
                new_batch = Batch(uuid.uuid4().hex, next_batch_reqs)
                self.finetuning_req_list = self.finetuning_req_list[len(next_batch_reqs) + aborted_count:]
                return new_batch
            else:
                return None