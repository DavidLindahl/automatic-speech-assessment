from pathlib import Path

p = Path('/work3/s234817/automatic-speech-assessment/jobs/sft/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3.sh')
src = Path('/work3/s234817/automatic-speech-assessment/jobs/sft/sft_warmup_plus3epoch_h100_ds_torchrun_retry.sh')
p.write_text(src.read_text())
text = p.read_text()
text = text.replace('#BSUB -J sft-warmup-plus3-h100-ds-tr', '#BSUB -J sft-warmup-plus3-h100-ds-tr-plus3')
text = text.replace('#BSUB -W 8:00', '#BSUB -W 12:00')
text = text.replace('logs/sft_warmup_plus3epoch_h100_ds_torchrun_retry_%J.out', 'logs/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3_%J.out')
text = text.replace('logs/sft_warmup_plus3epoch_h100_ds_torchrun_retry_%J.err', 'logs/sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3_%J.err')
text = text.replace('--model-name sft_warmup_plus3epoch_h100_ds_torchrun_retry', '--model-name sft_warmup_plus3epoch_h100_ds_torchrun_retry_plus3')
text = text.replace('"sft-warmup-plus3epoch-h100-ds-torchrun-retry"', '"sft-warmup-plus3epoch-h100-ds-torchrun-retry-plus3"')
p.write_text(text)
print(p)
