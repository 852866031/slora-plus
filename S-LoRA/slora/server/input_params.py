class ServeParams:

    def __init__(
        self,
        first_slo,
        token_slo,
    ) -> None:
        self.first_slo = first_slo
        self.token_slo = token_slo
        return
    
   
    def to_dict(self):
        ret = {}
        ret["first_slo"] = self.first_slo
        ret["token_slo"] = self.token_slo
        return ret


class FinetuneParams:
    def __init__(self, model_weightdir: str,
                 tokenizor_mode: str,
                 trust_remote_code: bool,
                 finetuning_config: dict):
        self.model_weightdir = model_weightdir
        self.tokenizor_mode = tokenizor_mode
        self.trust_remote_code = trust_remote_code
        if len(finetuning_config.keys())!=0:
            self.finetuning_type = finetuning_config.get("finetuning_type", "SFT")
            self.finetuning_data_path = finetuning_config.get("finetuning_data_path", None)
            self.finetuning_prepare_size = finetuning_config.get("finetuning_prepare_size", 0)
            self.finetuning_lora_path = finetuning_config.get("finetuning_lora_path", None)
            self.reference_lora_path = finetuning_config.get("reference_lora_path", None)
            self.learning_rate = finetuning_config.get("learning_rate", 1e-4)
            self.weight_decay = finetuning_config.get("weight_decay", 0.01)
            self.gamma = finetuning_config.get("gamma", 0.95)
            self.alpha = finetuning_config.get("alpha", 0.5)
            self.beta = finetuning_config.get("beta", 0.02)
            self.lambdas = finetuning_config.get("lambda", 2)
            self.eval_steps = finetuning_config.get("eval_steps", 100)
            self.num_epochs = finetuning_config.get("num_epochs", 1)
            self.max_saved_finetuning_tokens = finetuning_config.get("max_saved_finetuning_tokens", 512)
            self.max_finetuning_tokens_in_batch = finetuning_config.get("max_finetuning_tokens_in_batch", 256)
            self.optimizer_threading = finetuning_config.get("optimizer_threading", False)
            self.min_backward_sample_count = finetuning_config.get("min_backward_sample_count", 8)
            self.start_on_launch = finetuning_config.get("start_on_launch", True)

class InputParams:

    def __init__(
        self,
        max_req_total_len,
        # kv cache manager parameters
        max_total_token_num,
        pool_size_lora,
        batch_max_tokens,
        running_max_req_size,
        # mem_ratio,
        # adapter_ratio,
        # heuristic
        swap,
        prefetch,
        prefetch_size,
        scheduler,
        profile,
        batch_num_adapters,
        enable_abort,
        # kernel,
        # # debug
        dummy,
        no_lora_compute,
        no_lora_swap,
        # no_lora_copy,
        no_kernel,
        no_mem_pool,
        bmm,
        no_lora,
        # fairness
        fair_weights,
        # finetuning parameters
        model_weightdir,
        tokenizer_mode,
        trust_remote_code=True,
        finetuning_config = {}
    ) -> None:
        self.max_req_total_len = max_req_total_len
        self.max_total_token_num = max_total_token_num
        self.pool_size_lora = pool_size_lora
        self.batch_max_tokens = batch_max_tokens
        self.running_max_req_size = running_max_req_size
        # self.mem_ratio = mem_ratio
        # self.adapter_ratio = adapter_ratio

        self.swap = swap
        self.prefetch = prefetch
        self.prefetch_size = prefetch_size
        self.scheduler = scheduler
        self.profile = profile
        self.batch_num_adapters = batch_num_adapters
        self.enable_abort = enable_abort
        # self.kernel = kernel

        self.dummy = dummy
        self.no_lora_compute = no_lora_compute
        self.no_lora_swap = no_lora_swap
        # self.no_lora_copy = no_lora_copy
        self.no_kernel = no_kernel
        self.no_mem_pool = no_mem_pool
        self.bmm = bmm
        self.no_lora = no_lora
        
        self.finetuning_params = FinetuneParams(
            model_weightdir=model_weightdir,
            tokenizor_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            finetuning_config = finetuning_config
        )
        return
 
