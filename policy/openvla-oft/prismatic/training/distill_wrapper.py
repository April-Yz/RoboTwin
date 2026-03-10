"""Teacher-student wrapper for offline privileged distillation."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.future_projector import FutureObservationProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK


def unwrap_module(module: nn.Module) -> nn.Module:
    return getattr(module, "module", module)


class PrivilegedDistillWrapper(nn.Module):
    def __init__(
        self,
        vla: nn.Module,
        action_head: nn.Module,
        future_projector: nn.Module,
        *,
        proprio_projector: Optional[nn.Module] = None,
        teacher_detach: bool = True,
        use_proprio: bool = False,
        use_film: bool = False,
    ) -> None:
        super().__init__()
        self.vla = vla
        self.action_head = action_head
        self.future_projector = future_projector
        self.proprio_projector = proprio_projector
        self.teacher_detach = teacher_detach
        self.use_proprio = use_proprio
        self.use_film = use_film

        if not isinstance(unwrap_module(action_head), L1RegressionActionHead):
            raise ValueError("Privileged distillation currently supports only the L1 regression action head.")

    def _model_device(self) -> torch.device:
        return next(unwrap_module(self.vla).parameters()).device

    def _run_vla(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        additional_patch_embeddings: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        device = self._model_device()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                output_projector_features=True,
                proprio=batch["proprio"].to(device) if self.use_proprio and batch.get("proprio") is not None else None,
                proprio_projector=self.proprio_projector if self.use_proprio else None,
                use_film=self.use_film,
                additional_patch_embeddings=additional_patch_embeddings,
            )

    def _predict_actions_from_output(
        self,
        output: CausalLMOutputWithPast,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        projector_features = output.projector_features
        if projector_features is None:
            raise ValueError("Expected `output_projector_features=True` when running privileged distillation.")
        num_patch_tokens = projector_features.shape[1]

        current_action_mask = get_current_action_mask(labels)
        next_actions_mask = get_next_actions_mask(labels)
        all_actions_mask = current_action_mask | next_actions_mask

        last_hidden_states = output.hidden_states[-1]
        text_hidden_states = last_hidden_states[:, num_patch_tokens:-1]
        batch_size = labels.shape[0]
        actions_hidden_states = (
            text_hidden_states[all_actions_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1).to(torch.bfloat16)
        )
        return unwrap_module(self.action_head).predict_action(actions_hidden_states)

    def _get_language_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        vla = unwrap_module(self.vla)
        device = self._model_device()
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        input_embeddings = vla.get_input_embeddings()(input_ids)
        all_actions_mask = get_current_action_mask(labels) | get_next_actions_mask(labels)
        return input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

    def _encode_future_token(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "future_pixel_values" not in batch:
            raise ValueError("Privileged distillation requires `future_pixel_values` in the batch.")

        future_pixel_values = batch["future_pixel_values"]
        batch_size, future_horizon = future_pixel_values.shape[:2]
        device = self._model_device()
        vla = unwrap_module(self.vla)

        flattened_future_pixels = future_pixel_values.reshape(batch_size * future_horizon, *future_pixel_values.shape[2:])
        flattened_future_pixels = flattened_future_pixels.to(torch.bfloat16).to(device)

        language_embeddings = None
        if self.use_film:
            language_embeddings = self._get_language_embeddings(batch)
            language_embeddings = language_embeddings.repeat_interleave(future_horizon, dim=0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            future_patch_embeddings = vla._process_vision_features(
                flattened_future_pixels,
                language_embeddings=language_embeddings,
                use_film=self.use_film,
            )
            future_patch_embeddings = future_patch_embeddings.reshape(
                batch_size,
                future_horizon,
                future_patch_embeddings.shape[1],
                future_patch_embeddings.shape[2],
            )

            future_mask = batch.get("future_mask")
            if future_mask is not None:
                future_mask = future_mask.to(device)

            return unwrap_module(self.future_projector)(future_patch_embeddings, future_mask=future_mask)

    def forward_student(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output = self._run_vla(batch)
        labels = batch["labels"].to(self._model_device())[:, 1:]
        predicted_actions = self._predict_actions_from_output(output, labels)
        return {"output": output, "predicted_actions": predicted_actions}

    def forward_teacher(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        context = torch.no_grad() if self.teacher_detach else torch.enable_grad()
        with context:
            future_token = self._encode_future_token(batch)
            output = self._run_vla(batch, additional_patch_embeddings=future_token)
            labels = batch["labels"].to(self._model_device())[:, 1:]
            predicted_actions = self._predict_actions_from_output(output, labels)
        return {
            "output": output,
            "predicted_actions": predicted_actions,
            "future_token": future_token,
        }
