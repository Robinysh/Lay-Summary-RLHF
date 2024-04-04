import os
from collections import defaultdict

import bitsandbytes as bnb
import lightning as L
import torch
import trl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from peft import LoraConfig
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from unsloth import FastLanguageModel

from burrito.metrics_ppo.dataloader import (
    AbstractDataModule,
    AbstractDataset,
    decoder_collate,
)
from burrito.metrics_ppo.rewarder import Rewarder
from burrito.metrics_ppo.utils import time_it

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("medium")

# torch.autograd.set_detect_anomaly(True)

ARTICLES_FPATH = "/data/colx531/biolaysumm2024_data"
LOG_FREQ = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_RANK = 32
# MODEL_NAME = "stabilityai/stablelm-2-zephyr-1_6b"
MODEL_NAME = "BioMistral/BioMistral-7B-DARE"
# MODEL_NAME = "M4-ai/tau-0.5B"
BATCH_SIZE = 1


# pylint: disable-next=too-many-instance-attributes
class PPOLM(L.LightningModule):
    def __init__(self, model, tokenizer, lr=1e-4):
        super().__init__()
        self.model = model
        self.val_step_outs = defaultdict(list)
        self.rewarder = Rewarder()
        self.lr = lr
        self.tokenizer = tokenizer
        self.automatic_optimization = False

        self.gen_kwargs = {
            "min_new_tokens": 50,
            "max_new_tokens": 700,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        # output_min_length = 4
        # output_max_length = 16
        # output_length_sampler = LengthSampler(output_min_length, output_max_length)

    def on_fit_start(self) -> None:
        config = PPOConfig(
            model_name=MODEL_NAME,
            learning_rate=self.lr,
            log_with="wandb",
            batch_size=BATCH_SIZE,
            mini_batch_size=BATCH_SIZE,
            optimize_device_cache=True,
        )

        # pylint: disable-next=attribute-defined-outside-init
        self.ppo_trainer = PPOTrainer(
            config,
            self.model,
            tokenizer=self.tokenizer,
            optimizer=self.optimizers(),
        )
        return super().on_fit_start()

    # pylint: disable-next=arguments-differ
    def training_step(self, batch, batch_idx):
        inputs, _, encoded_sents, queries, articles, article_ids = batch

        with time_it("generate"):
            with torch.no_grad():
                outputs = self.ppo_trainer.generate(
                    encoded_sents,
                    return_prompt=False,
                    batch_size=BATCH_SIZE,
                    # use_cache=False,
                    **self.gen_kwargs,
                )

        with time_it("decode"):
            decoded_seqs = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )

        if batch_idx % LOG_FREQ == 0:
            self.logger.experiment.log(
                {
                    "train/generate_predicted": decoded_seqs[0],
                    "train/generate_answer": queries[0],
                    "train/filename": article_ids[0],
                }
            )

        with time_it("reward"):
            reward, score_dict = self.rewarder(decoded_seqs, articles)
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        self.log_dict({f"train/{k}": sum(v) / len(v) for k, v in score_dict.items()})
        self.log("train/reward", reward.mean())

        with time_it("ppo_step"):
            stats = self.ppo_trainer.step(
                encoded_sents, outputs, list(torch.split(reward, 1))
            )

        self.ppo_trainer.log_stats(
            stats, inputs | {"response": decoded_seqs, "query": queries}, reward
        )

    # pylint: disable-next=arguments-differ
    def validation_step(self, batch, batch_idx):
        _, _, encoded_sents, queries, articles, article_ids = batch

        outputs = self.ppo_trainer.generate(
            encoded_sents, return_prompt=False, **self.gen_kwargs
        )

        decoded_seqs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if batch_idx % LOG_FREQ == 0:
            self.logger.experiment.log(
                {
                    "val/generate_predicted": decoded_seqs[0],
                    "val/generate_answer": queries[0],
                    "val/filename": article_ids[0],
                }
            )

        reward, score_dict = self.rewarder(decoded_seqs, articles)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        self.log_dict({f"val/{k}": sum(v) / len(v) for k, v in score_dict.items()})
        self.log("val/reward", reward.mean())

    def configure_optimizers(self):
        optimizer = bnb.optim.Adam8bit(self.parameters(), lr=self.lr)
        return optimizer


def get_model(model_name, unsloth=False):
    if unsloth:

        def get_unsloth_model(base_model_name):
            model, _ = FastLanguageModel.from_pretrained(
                model_name=base_model_name,
                max_seq_length=2048,
                load_in_4bit=True,
                # quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            return FastLanguageModel.get_peft_model(
                model,
                target_modules=[
                    "q_proj",
                    "v_proj",
                    "k_proj",
                    "o_proj",  # attention (self_attn)
                    "gate_proj",
                    "down_proj",
                    "up_proj",  # FFN (mlp)
                ],
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                # lora_dropout = LORA_DROPOUT,
                lora_dropout=0,
                bias="none",
                use_gradient_checkpointing=False,
            )

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            get_unsloth_model(model_name)
        )
        # model.gradient_checkpointing_disable()
        trl.trainer.peft_module_casting_to_bf16(model)
        return model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    peft_config = LoraConfig(
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        r=LORA_RANK,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        peft_config=peft_config,
    )
    model.gradient_checkpointing_disable()
    return model


def main():
    model = get_model(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, pad_token="</s>"
    )
    tokenizer.pad_token = tokenizer.eos_token

    wandb_logger = WandbLogger(project="COLX531")
    checkpoint_callback = ModelCheckpoint(
        dirpath="/data3/robinysh/models/colx531",
        filename="{epoch}-{val/acc:.2f}",
        save_top_k=1,
        monitor="val/loss",
        every_n_epochs=1,
        mode="max",
    )

    dm = AbstractDataModule(
        articles_fpath=ARTICLES_FPATH,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        dataset=AbstractDataset,
        collate_fn=decoder_collate,
    )
    model = PPOLM(model=model, tokenizer=tokenizer, lr=1e-6)
    trainer = L.Trainer(
        precision="bf16-true",
        accelerator="cuda",
        logger=wandb_logger,
        val_check_interval=256,
        limit_val_batches=10,
        log_every_n_steps=LOG_FREQ,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
