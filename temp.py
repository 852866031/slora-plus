def _context_ffn(self, input, layer_weight): # the function compute ffn
    input1 = self._ffn_norm(input, layer_weight)
    ffn_out = self._ffn(input1, layer_weight)
    input1 = None
    if self.world_size_ > 1:
        dist.all_reduce(ffn_out, op=dist.ReduceOp.SUM, async_op=False)
    input.add_(ffn_out.view(-1, self.embed_dim_))
    return

def _ffn_norm(self, input, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
    return rmsnorm_forward(input, weight=layer_weight.ffn_norm_weight_, eps=self.eps_)

def _ffn(self, input, layer_weight:LlamaTransformerLayerWeight)->torch.Tensor:
        gate_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.gate_proj)
        torch.nn.functional.silu(gate_out, inplace=True)
        up_out = torch.mm(input.view(-1, self.embed_dim_), layer_weight.up_proj)
        input = None
        ffn1_out = gate_out * up_out
        gate_out, up_out = None, None
        ffn2_out = torch.mm(ffn1_out, layer_weight.down_proj)
        ffn1_out = None
        return ffn2_out