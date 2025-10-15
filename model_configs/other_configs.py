"""
Use this file to define configs for baselines
"""

def set_model_config(args):
    config = {}
    ValueError(f"Configs for {args.mil_name} not found")
    return config

def set_genomic_config(args):
    if args.geneclass_name == "gene_mixer_group":
        config = {
            "latent_dim": 256,
            "depth": 3,
            "expansion_groups": 0.5,
            "expansion_dim": 0.5,
            "dropout": 0.25,
            "cls_token": False,
            "n_classes": args.num_classes,
            "final_groups": 64,
        }
    else:
        raise ValueError(f"Configs for {args.geneclass_name} not found")
    return config
