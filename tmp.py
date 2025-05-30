def _lora_get_qkv(self, layer_id, input_embs, cache_k, cache_v, infer_state, no_lora_compute=False)->torch.Tensor:
        # current_w_combined_A = self.get_w_combined_A(
        #     self.infer_adapter.a_loc, self.infer_adapter.a_start, self.infer_adapter.a_len, 0, layer_id)
        # print(f"hashed current_w_combined_A for layer {layer_id}, {self.hash_tensor(current_w_combined_A)}")
        base_model = self.base_model
        base_layer_weight = base_model.trans_layers_weight[layer_id]
        base_layer_infer = base_model.layers_infer[layer_id]
        # q (S, H)
        q = torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_),
                     base_layer_weight.q_weight_)
        assert(len(q)==len(self.batch_req_bins))
        # q = q_base + input * A * B * scaling
        # input: (S, H) A: (H, R) B: (R, H)
        # fix me: @TODO we need to filter out requests querying only base model
        delta_qA = self.delta[0]
        dispatch_bgmv(delta_qA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                    self.key_buffer[layer_id],
                    self.infer_adapter.a_start, self.infer_adapter.a_len, 
                    self.infer_adapter.a_loc, self.batch_req_bins, 0, self.infer_adapter.a_scaling)
        dispatch_bgmv(q, delta_qA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                    self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                    self.batch_req_bins, 0, self.infer_adapter.a_scaling)
        rotary_emb_fwd(q.view(-1, base_layer_infer.tp_q_head_num_, base_model.head_dim_),
                    infer_state.position_cos, infer_state.position_sin)

        # k (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.k_weight_,
                 out=cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        dispatch_bgmv(delta_kA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                    self.key_buffer[layer_id], 
                    self.infer_adapter.a_start, self.infer_adapter.a_len, 
                    self.infer_adapter.a_loc, self.batch_req_bins, 1, self.infer_adapter.a_scaling)
        dispatch_bgmv(cache_k.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                    delta_kA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                    self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                    self.batch_req_bins, 1, self.infer_adapter.a_scaling)
        # delta_kA = None
        rotary_emb_fwd(cache_k, infer_state.position_cos, infer_state.position_sin)

        # v (S, H)
        torch.mm(input_embs.view(-1, base_layer_infer.embed_dim_), base_layer_weight.v_weight_,
                 out=cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_))
        dispatch_bgmv(delta_vA, input_embs.view(-1, base_layer_infer.embed_dim_), 
                    self.key_buffer[layer_id], 
                    self.infer_adapter.a_start, self.infer_adapter.a_len, 
                    self.infer_adapter.a_loc, self.batch_req_bins, 2, self.infer_adapter.a_scaling)
        dispatch_bgmv(cache_v.view(-1, base_model.tp_k_head_num_ * base_model.head_dim_), 
                    delta_vA, self.value_buffer[layer_id], self.infer_adapter.a_start, 
                    self.infer_adapter.a_len, self.infer_adapter.a_loc, 
                    self.batch_req_bins, 2, self.infer_adapter.a_scaling)
            # delta_vA = None
        return q, q_base