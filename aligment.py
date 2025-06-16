import copy, torch
from transformers import GPT2Config, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model

def tiny_gpt2_backbone() -> GPT2LMHeadModel:
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=128, vocab_size=100)
    return GPT2LMHeadModel(cfg).half().cuda()

def add_trainable_lora(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    lora_cfg = LoraConfig(r=4, lora_alpha=8, target_modules=["c_attn", "c_proj"])
    return get_peft_model(model, lora_cfg)

def frozen_copy(model: GPT2LMHeadModel) -> GPT2LMHeadModel:
    ref = copy.deepcopy(model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref

def make_fake_batch(batch_size: int = 2,
                    seq_len:    int = 8,
                    vocab:      int = 100,
                    device:     str = "cuda"):
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, seq_len // 2 :] = True  # assume half of the sequence is completion 
    label = torch.randint(0, 2, (batch_size,), device=device, dtype=torch.int8)
    label = label * 2 - 1  # convert to {-1, 1} labels which stands for "good completion" and "bad completion"
    return x, mask, label

def seq_logp(model_out, input_ids, compl_mask):
    logits = model_out.logits
    logp   = torch.log_softmax(logits, -1)
    tgt    = input_ids[:, 1:]
    mask   = compl_mask[:, 1:]
    tok_lp = torch.gather(logp[:, :-1], 2, tgt.unsqueeze(-1)).squeeze(-1)
    return (tok_lp * mask).sum(-1)           # shape (batch,)

def kto_loss(logp_pol, logp_ref, label, alpha=0.25, beta=0.02):
    delta = logp_pol - logp_ref              
    w     = torch.where(label == 1, alpha, 1.0)   
    return (-w * delta + beta * delta.detach()).mean()

def main():
    base      = tiny_gpt2_backbone()
    policy    = add_trainable_lora(base)    
    reference = frozen_copy(policy)       
    optim = torch.optim.Adam(policy.parameters(), lr=1e-4)

    x, mask, label = make_fake_batch()

    with torch.no_grad():                    
        logp_ref = seq_logp(reference(x), x, mask)

    logp_pol = seq_logp(policy(x), x, mask) 
    loss     = kto_loss(logp_pol, logp_ref, label)
    print(loss)

    loss.backward()                        
    optim.step(); optim.zero_grad()

    print(f"KTO step completed   loss={loss.item():.4f}")

if __name__ == "__main__":
    main()