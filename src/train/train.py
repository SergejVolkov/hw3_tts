import torch
from torch import nn

import os
from tqdm import tqdm

from src.inference import inference


def train(train_config, training_loader, model, optimizer, scheduler,
          fastspeech_loss, logger, waveglow_model, current_step = 0):
    total_steps = train_config.epochs * len(training_loader) * \
                  train_config.batch_expand_size
    tqdm_bar = tqdm(total=total_steps - current_step)

    n_checkpoints = 0
    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)
                if current_step > total_steps:
                    break

                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                pitch = db["pitch"].float().to(train_config.device)
                energy = torch.sqrt((mel_target.exp() ** 2).mean(dim=-1))
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                (mel_output, duration_predictor_output, pitch_predictor_output,
                 energy_predictor_output) = model(character,
                                                  src_pos,
                                                  mel_pos=mel_pos,
                                                  mel_max_length=max_mel_len,
                                                  length_target=duration,
                                                  pitch_target=pitch,
                                                  energy_target=energy)

                # Calc Loss
                (mel_loss, duration_loss, pitch_loss,
                 energy_loss) = fastspeech_loss(mel_output,
                                                duration_predictor_output,
                                                pitch_predictor_output,
                                                energy_predictor_output,
                                                mel_target,
                                                duration,
                                                pitch,
                                                energy)
                total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()
                p_l = pitch_loss.detach().cpu().numpy()
                e_l = energy_loss.detach().cpu().numpy()

                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("pitch_loss", p_l)
                logger.add_scalar("energy_loss", e_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    if (train_config.max_checkpoints is not None and
                            n_checkpoints >= train_config.max_checkpoints):
                        os.remove(os.path.join(
                            train_config.checkpoint_path,
                            "checkpoint_%d.pth.tar" %
                            (current_step - train_config.save_step * n_checkpoints)
                        ))
                    else:
                        n_checkpoints += 1

                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()
                    }, os.path.join(train_config.checkpoint_path,
                                    "checkpoint_%d.pth.tar" % current_step))

                    model.eval()
                    result = inference(train_config.tests_path,
                                       model, waveglow_model,
                                       train_config.text_cleaners,
                                       train_config.device)
                    model.train()
                    for i, (audio, mel) in enumerate(zip(*result)):
                        logger.add_audio(f"a_sample{i}", audio.detach().cpu().short(), sample_rate=22050)
                        logger.add_image(f"mel_sample{i}", mel.detach().cpu().numpy()[::-1])

    torch.save({
        "model": model.state_dict()
    }, os.path.join(train_config.checkpoint_path,
                    "final_%d.pth.tar" % current_step))
