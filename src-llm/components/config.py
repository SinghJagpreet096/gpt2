GPT_CONFIG_124M = {
    "vocab_size": 50257,  # vocabulary size  (default 50257)
    "batch_size": 12,  # batch size
    "ctx_len": 32,  # context length (default 1024)
    "emb_dim": 768,  # embedding dimesension (default 768)
    "n_head": 12,  # number of attention heads (default 12)
    "n_layers": 12,  # number of layers (default 12)
    "learning_rate": 6e-4,  # learning rate
    "drop_rate": 0.1,  # Dropout rate
    "qkv_bias": False,  # qkv bias
    "eval_interval": 100,  # evaluation interval (default 200)
    "max_iters": 1000,  # maximum iterations (default 60000)
    "eval_iters": 200,
}
