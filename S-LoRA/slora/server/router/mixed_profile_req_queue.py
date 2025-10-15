import copy
import random
import string
import time
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

max_new_tokens_default = 40
inference_requests_length_default = 50


# 50 clients, 3 reqs --> flood the receiver
# 1 client 150 --> smoother
# generate_requests
# 0.3s slower in the co-serving

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


def get_inference_sampling_params(max_new_tokens) -> SamplingParams:
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
        ignore_eos=True,     # Typically handle EOS in training
        max_new_tokens=max_new_tokens,   # Arbitrary placeholder
        stop_sequences=[]
    )
    return sp


def generate_random_sentence(length: int) -> str:
    """Generate a random sentence of `length` words."""
    words = []
    for _ in range(max(1, length)):
        word = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        words.append(word)
    return " ".join(words).capitalize() + "."

def generate_inference_req(adapter_dir: str, length: int, max_new_tokens: int, tokenizer):
    prompt_text = generate_random_sentence(length)
    prompt_ids = tokenizer(prompt_text).get('input_ids', [])
    req  = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=prompt_ids,
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,  
        needs_to_notify_detokenize = True,           
        text=prompt_text,  
    )
    return req

def generate_dual_inference_req(adapter_dir: str, length: int, max_new_tokens: int, tokenizer):
    prompt_text = generate_random_sentence(length)
    prompt_ids = tokenizer(prompt_text).get('input_ids', [])
    req_1 = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=prompt_ids,
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,     
        needs_to_notify_detokenize = True,         
        text=prompt_text,  
    )
    req_2 = Req(
        adapter_dir=adapter_dir,                
        request_id=uuid.uuid4().hex,     
        prompt_ids=copy.deepcopy(prompt_ids),
        sample_params=get_inference_sampling_params(max_new_tokens),
        is_finetuning=False,              
        needs_to_notify_detokenize = True,
        text=copy.deepcopy(prompt_text),  
    )
    return req_1, req_2

class MixedProfile_ReqQueue:
    """
    # first wave: warmup   
    # second wave: inference only
    # third wave: mixed inference + finetuning
    # Inference requests in second wave and third wave are identical
    """

    def __init__(self,
                 max_total_tokens: int,
                 batch_max_tokens: int,
                 running_max_req_size: int,
                 finetune_params: FinetuneParams,
                 finetuning_adapters_tracker,
                 inference_adapter_dir: str):
        self.max_total_tokens = max_total_tokens #1024
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        self.inference_adapter_dir = inference_adapter_dir
        print(f"max_total_tokens: {self.max_total_tokens}")
        print(f"batch_max_tokens: {self.batch_max_tokens}")
        print(f"running_max_req_size: {self.running_max_req_size}")
        print(f"inference_adapter_dir: {inference_adapter_dir}")
        # config parameters
        self.finetuning_data_path = finetune_params.finetuning_data_path
        self.finetuning_prepare_size = finetune_params.finetuning_prepare_size
        self.finetuning_lora_path = finetune_params.finetuning_lora_path  
        self.max_saved_finetuning_tokens = finetune_params.max_saved_finetuning_tokens  #max size of saved activations in memory
        self.max_finetuning_tokens_in_batch = finetune_params.max_finetuning_tokens_in_batch #max size of finetuning tokens in a forward batch
        self.total_epoch = finetune_params.num_epochs
        self.finetuning_adapters_tracker = finetuning_adapters_tracker
        self.start_task= finetune_params.start_on_launch
        # tracker variables
        self.finetuning_tokens_in_memory = 0 #
        self.finetuning_tokens_processed = 0
        self.pending_bwd_tokens = 0
        self.epoch_avg_loss_list = []
        self.loss_list =[]
        self.current_epoch = 0
        self.sample_index = 0
        self.last_index = 0
        self.flag = True
        
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
        print("Initializing Finetuning Settings")
        print(f"Finetuning data path: {self.finetuning_data_path}")
        print(f"Finetuning lora path: {self.finetuning_lora_path}")
        print(f"Finetuning prepare size: {self.finetuning_prepare_size}")
        print(f"Max saved finetuning tokens: {self.max_saved_finetuning_tokens}")
        print(f"batch max tokens: {self.batch_max_tokens}")
        self.prepare_finetuning_requests()
        self.warmup_request_list = self.prepare_warmup_requests()
        self.first_wave_inf, self.second_wave_inf = self.prepare_inference_requests()

    def prepare_warmup_requests(self):
        warmup_request_list = []
        for _ in range(5):
            req = generate_inference_req(
                adapter_dir=self.inference_adapter_dir,
                length=10,
                max_new_tokens=10,
                tokenizer=self.tokenizer
            )
            warmup_request_list.append(req)
        return warmup_request_list

    def prepare_inference_requests(self):
        inference_only_requests_list = []
        mixed_requests_list = []
        for _ in range(3):
            req_1, req_2 = generate_dual_inference_req(
                adapter_dir=self.inference_adapter_dir,
                length=inference_requests_length_default,
                max_new_tokens=max_new_tokens_default,
                tokenizer=self.tokenizer
            )
            inference_only_requests_list.append(req_1)
            mixed_requests_list.append(req_2)
        return inference_only_requests_list, mixed_requests_list

    def is_heavy_loading(self):
        #TODO : implement a better way to check if the queue is heavy loading
        if len(self.waiting_req_list) > 100:
            return True
        else:
            return False


    def update_finetuning_status_after_fwd(self, batch: Batch):
        for req in batch.reqs:
            if req.is_finetuning:
                self.pending_bwd_tokens += req.input_len

    def update_finetuning_status_after_bwd(self, loss_list, num_processed_tokens):
        self.loss_list.extend(loss_list)
        self.finetuning_tokens_processed += num_processed_tokens
        self.pending_bwd_tokens -= num_processed_tokens
        #print bar
        bar_width = 50
        ratio = self.finetuning_tokens_processed / max(self.finetuning_tokens_in_memory, 1)
        filled_len = int(bar_width * ratio)
        empty_len = bar_width - filled_len
        grey = "*"
        white = " "
        bar = grey * filled_len + white * empty_len
        print(f"Epoch: {self.current_epoch+1}/{self.total_epoch} [{bar}] {ratio:.1%} processed", flush=True)
        if self.finetuning_tokens_processed >= self.finetuning_tokens_in_memory:
            self.epoch_avg_loss_list.append(np.mean(self.loss_list))
            print(f"Epoch {self.current_epoch} finished. Average Loss: {self.epoch_avg_loss_list[-1]:.6f}\n")
            for i, loss in enumerate(self.loss_list):
                print(f"Epoch {self.current_epoch} Batch {i}: Loss = {loss:.6f}")
            self.loss_list = []
            self.current_epoch += 1
            if self.current_epoch >= self.total_epoch:
                print("=== Loss List ===")
                for i, loss in enumerate(self.epoch_avg_loss_list):
                    print(f"Backward Epoch {i}: Loss = {loss:.6f}")
                print("=== End of Loss List ===", flush=True)
            else:
                self.sample_index = 0
                self.finetuning_tokens_processed = 0
        else:
            print()

    def finetuning_is_finished(self, sender=None):
        if self.current_epoch >= self.total_epoch:
            if sender is not None:
                sender.send_pyobj(True)
            return True
        else:
            return False
        

    def append(self, req: Req):
        self.waiting_req_list.append(req)

    def prepare_finetuning_requests(self):
        loaded_count = 0
        try:
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
                    self.finetuning_req_list.append(new_req)
                    self.finetuning_tokens_in_memory+=new_req.input_len
                    loaded_count += 1
                    if loaded_count >= self.finetuning_prepare_size:
                        break
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
    
    def print_new_batch_info(self, can_run_list):
        if len(can_run_list) > 0:
            infer_tokens = 0
            finetune_tokens = 0
            for req in can_run_list:
                if req.is_finetuning:
                    finetune_tokens += req.input_len
                else:
                    infer_tokens += req.input_len
            unused = self.batch_max_tokens - (infer_tokens + finetune_tokens)
            print(f"\033[34mForward Batch Token Layout: [{infer_tokens} infer tokens/ {finetune_tokens} finetune (max: {self.max_finetuning_tokens_in_batch}) / {unused} unused] \033[0m")
            print(f"\033[34mPending BWD token count: {self.pending_bwd_tokens}\033[0m")
        

    def generate_new_batch(self, current_batch: Optional[Batch], lora_ranks: dict[str, int]) -> Optional[Batch]:
        if current_batch is not None:
            return None
        
        self._init_cache_list(current_batch, lora_ranks)
        new_batch_total_tokens = 0
        can_run_list = []
        aborted_count = 0
        # 1) If the warmup requests are not finished, run them first
        if len(self.warmup_request_list) > 0:
            for req in self.warmup_request_list:
                if req.aborted:
                    print("Request aborted")
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            if len(can_run_list) > 0:
                self.warmup_request_list = self.warmup_request_list[len(can_run_list) + aborted_count:]
            self.print_new_batch_info(can_run_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            print(f"\033[32mWarmup Batch Created with {len(can_run_list)} requests\033[0m")
            return new_batch
        elif len(self.first_wave_inf) > 0:
            for req in self.first_wave_inf:
                if req.aborted:
                    print("Request aborted")
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            if len(can_run_list) > 0:
                self.first_wave_inf = self.first_wave_inf[len(can_run_list) + aborted_count:]

            #self.add_finetuning_req(new_batch_total_tokens, can_run_list)

            self.print_new_batch_info(can_run_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            print(f"\033[32mFirst Wave Inference Batch Created with {len(can_run_list)} requests\033[0m")
            return new_batch
        elif len(self.second_wave_inf) > 0:
            for req in self.second_wave_inf:
                if req.aborted:
                    print("Request aborted")
                    aborted_count += 1
                    continue
                if (self._can_add_new_req(req, lora_ranks) and
                    (new_batch_total_tokens + req.input_len) <= self.batch_max_tokens):
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                else:
                    break
            if len(can_run_list) > 0:
                self.second_wave_inf = self.second_wave_inf[len(can_run_list) + aborted_count:]

                
            self.add_finetuning_req(new_batch_total_tokens, can_run_list)

            self.print_new_batch_info(can_run_list)
            new_batch = Batch(uuid.uuid4().hex, can_run_list)
            print(f"\033[32mSecond Wave Mixed Batch Created with {len(can_run_list)} requests\033[0m")
            return new_batch
        elif self.flag:
            print("All profiling samples done")
            self.flag = False
            return None
        else:
            return None

    def add_finetuning_req(self, new_batch_total_tokens, can_run_list):
        new_batch_inference_tokens=new_batch_total_tokens
        if len(self.finetuning_req_list)> 0 and self.finetuning_adapters_tracker.all_adapters_available():
            new_batch_finetuning_tokens = 0
            self.last_index = self.sample_index
            for i in range(self.sample_index, len(self.finetuning_req_list)):
                req = self.finetuning_req_list[i]
                if new_batch_total_tokens + req.input_len > self.batch_max_tokens:
                    break
                if new_batch_finetuning_tokens + req.input_len > self.max_finetuning_tokens_in_batch:
                    break
                if self.pending_bwd_tokens + new_batch_finetuning_tokens + req.input_len > self.max_saved_finetuning_tokens:
                    break
                if new_batch_inference_tokens!= 0 and new_batch_finetuning_tokens + req.input_len > new_batch_inference_tokens*2:
                    break
                else:
                    can_run_list.append(req)
                    new_batch_total_tokens += req.input_len
                    new_batch_finetuning_tokens += req.input_len
                    self.sample_index += 1
                    break

    def next_batch(self) -> Optional[Batch]:
       print("##### NEXT BATCH CALLED #####")
       return None
    

    def update_counter(self, req: Req):
        pass