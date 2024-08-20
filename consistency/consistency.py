import copy
import math
import os
from contextlib import suppress
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
from diffusers import UNet2DModel
from diffusers.models.unet_2d import UNet2DOutput
from diffusers.utils.torch_utils import randn_tensor
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import nn, optim
from torchmetrics import MeanMetric
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

from consistency.pipeline import ConsistencyPipeline

with suppress(ImportError):
    import wandb


class Consistency(LightningModule):
    def __init__(
        self,
        model: UNet2DModel,
        *,
        loss_fn: nn.Module = nn.MSELoss(),
        learning_rate: float = 1e-4,
        data_std: float = 0.5,
        time_min: float = 0.002,
        time_max: float = 80.0,
        bins_min: int = 2,
        bins_max: int = 150,
        bins_rho: float = 7,#小区间有几个
        initial_ema_decay: float = 0.9,
        optimizer_type: Type[optim.Optimizer] = optim.RAdam,
        samples_path: str = "samples/",
        save_samples_every_n_epoch: int = 10,
        num_samples: int = 16,
        sample_steps: int = 1,
        use_ema: bool = True,
        sample_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.model_ema = copy.deepcopy(model)
        self.image_size = model.sample_size
        self.channels = model.in_channels

        self.model_ema.requires_grad_(False)

        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate
        self.initial_ema_decay = initial_ema_decay

        self.data_std = data_std
        self.time_min = time_min
        self.time_max = time_max
        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho

        self._loss_tracker = MeanMetric()
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()

        Path(samples_path).mkdir(exist_ok=True, parents=True)

        self.samples_path = samples_path
        self.save_samples_every_n_epoch = save_samples_every_n_epoch
        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.use_ema = use_ema
        self.sample_seed = sample_seed

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
    ):
        return self._forward(self.model, images, times)
    def _forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        times: torch.Tensor,
        clip: bool = True,
    ):  

        skip_coef = self.data_std**2 / (
            (times - self.time_min).pow(2) + self.data_std**2
        )

        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        out: UNet2DOutput = model(images, times)

        out = self.image_time_product(
            images,
            skip_coef,
        ) + self.image_time_product(
            out.sample,
            out_coef,
        )

        if clip:
            return out.clamp(-1.0, 1.0)

        return out 

    def training_step(self, images: torch.Tensor, *args, **kwargs):
        _bins = self.bins

        noise = torch.randn(images.shape, device=images.device)
        timesteps = torch.randint(
            0,
            _bins - 1,
            (images.shape[0],),
            device=images.device,
        ).long()

        current_times = self.timesteps_to_times(timesteps, _bins)
        next_times = self.timesteps_to_times(timesteps + 1, _bins)

        current_noise_image = images + self.image_time_product(
            noise,
            current_times,
        )

        next_noise_image = images + self.image_time_product(
            noise,
            next_times,
        )

        with torch.no_grad():
            target = self._forward(
                self.model_ema,
                current_noise_image,
                current_times,
            )

        loss = self.loss_fn(self(next_noise_image, next_times), target)

        self._loss_tracker(loss)
        self.log(
            "loss",
            self._loss_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        self._bins_tracker(_bins)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), lr=self.learning_rate)             
    

    

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)
        self.ema_update()

    @torch.no_grad()
    def ema_update(self):
        param = [p.data for p in self.model.parameters()]
        param_ema = [p.data for p in self.model_ema.parameters()]

        torch._foreach_mul_(param_ema, self.ema_decay)
        torch._foreach_add_(param_ema, param, alpha=1 - self.ema_decay)

        self._ema_decay_tracker(self.ema_decay)
        self.log(
            "ema_decay",
            self._ema_decay_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    @property
    def ema_decay(self):
        return math.exp(self.bins_min * math.log(self.initial_ema_decay) / self.bins)

    @property
    def bins(self) -> int:
        return math.ceil(
            math.sqrt(
                self.trainer.global_step
                / self.trainer.estimated_stepping_batches
                * (self.bins_max**2 - self.bins_min**2)
                + self.bins_min**2
            )
        )
    


    def timesteps_to_times(self, timesteps: torch.LongTensor, bins: int):
        return (
            (
                self.time_min ** (1 / self.bins_rho)
                + timesteps
                / (bins - 1)
                * (
                    self.time_max ** (1 / self.bins_rho)
                    - self.time_min ** (1 / self.bins_rho)
                )
            )
            .pow(self.bins_rho)
            .clamp(0, self.time_max)
        )

    @rank_zero_only
    def on_train_start(self) -> None:
        self.save_samples(
            f"{0:05}",
            num_samples=self.num_samples,
            steps=self.sample_steps,
            generator=torch.Generator(device=self.device).manual_seed(self.sample_seed),
            use_ema=self.use_ema,
        )

    @rank_zero_only
    def on_train_epoch_end(self) -> None:
        if (
            (self.trainer.current_epoch < 30)
            or ((self.trainer.current_epoch + 1) % self.save_samples_every_n_epoch == 0)
            or self.trainer.current_epoch == (self.trainer.max_epochs - 1)
        ):
            self.save_samples(
                f"{(self.current_epoch+1):05}",
                num_samples=self.num_samples,
                steps=self.sample_steps,
                generator=torch.Generator(device=self.device).manual_seed(
                    self.sample_seed
                ),
                use_ema=self.use_ema,
            )

    @torch.no_grad()
    def sample(
        self,
        num_samples: int = 16,
        steps: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        use_ema: bool = False,
    ) -> torch.Tensor:
        shape = (num_samples, self.channels, self.image_size, self.image_size)

        time = torch.tensor([self.time_max], device=self.device)

        images: torch.Tensor = self._forward(
            self.model_ema if use_ema else self.model,
            randn_tensor(shape, generator=generator, device=self.device) * time,
            time,
        )

        if steps <= 1:
            return images

        _timesteps = list(
            reversed(range(0, self.bins_max, self.bins_max // steps - 1))
        )[1:]
        _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

        times = self.timesteps_to_times(
            torch.tensor(_timesteps, device=self.device), bins=150
        )

        for time in times:
            noise = randn_tensor(shape, generator=generator, device=self.device)
            images = images + math.sqrt(time.item() ** 2 - self.time_min**2) * noise
            images = self._forward(
                self.model_ema if use_ema else self.model,
                images,
                time[None],
            )

        return images

    @torch.no_grad()
    def save_samples(
        self,
        filename: str,
        num_samples: int = 16,
        steps: int = 1,
        use_ema: bool = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    ):
        samples = self.sample(
            num_samples=num_samples,
            steps=steps,
            generator=generator,
            use_ema=use_ema,
        )
        samples.mul_(0.5).add_(0.5)
        grid = make_grid(
            samples,
            nrow=math.ceil(math.sqrt(samples.size(0))),
            padding=self.image_size // 16,
        )

        save_image(
            grid,
            f"{self.samples_path}/{filename}.png",
            "png",
        )

        if isinstance(self.trainer.logger, WandbLogger):
            wandb.log(
                {
                    "samples": wandb.Image(to_pil_image(grid)),
                },
                commit=False,
                step=self.trainer.global_step,
            )

        del samples
        del grid
        torch.cuda.empty_cache()

    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)
    



class ImprovedConsistency(LightningModule):
    def __init__(
        self,
        model: UNet2DModel,
        *,
        loss_fn: nn.Module = nn.MSELoss(),#这里还需要一个c作为正则量
        learning_rate: float = 1e-4,
        data_std: float = 0.5,
        time_min: float = 0.002,
        time_max: float = 80.0,
        bins_min: int = 10,
        bins_max: int = 1280,
        P_mean = -1.1,
        P_std = 2.0,
        bins_rho: float = 7,
        optimizer_type: Type[optim.Optimizer] = optim.RAdam,
        samples_path: str = "samples/",
        save_samples_every_n_epoch: int = 10,
        num_samples: int = 16,
        sample_steps: int = 1,
        sample_seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.model=model
        self.image_size = model.sample_size
        self.channels = model.in_channels


        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type

        self.learning_rate = learning_rate

        self.data_std = data_std
        self.time_min = time_min
        self.time_max = time_max
        self.bins_min = bins_min
        self.bins_max = bins_max
        self.bins_rho = bins_rho
        self.P_mean = P_mean
        self.P_std = P_std

        self._loss_tracker = MeanMetric()
        self._bins_tracker = MeanMetric()
        self._ema_decay_tracker = MeanMetric()

        Path(samples_path).mkdir(exist_ok=True, parents=True)

        self.samples_path = samples_path
        self.save_samples_every_n_epoch = save_samples_every_n_epoch
        self.num_samples = num_samples
        self.sample_steps = sample_steps
        self.sample_seed = sample_seed

    def forward(
        self,
        images: torch.Tensor,
        times: torch.Tensor,
    ):
        return self._forward(self.model, images, times)
    def _forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        times: torch.Tensor,
        clip: bool = True,
    ):  

        skip_coef = self.data_std**2 / (
            (times - self.time_min).pow(2) + self.data_std**2
        )

        out_coef = self.data_std * times / (times.pow(2) + self.data_std**2).pow(0.5)

        out: UNet2DOutput = model(images, times)

        out = self.image_time_product(
            images,
            skip_coef,
        ) + self.image_time_product(
            out.sample,
            out_coef,
        )

        if clip:
            return out.clamp(-1.0, 1.0)

        return out 
    
    def training_step(self, images: torch.Tensor, *args, **kwargs):
        _bins = self.bins

        noise = torch.randn(images.shape, device=images.device)
        sigmas = self.karras_schedule(_bins, device=images.device)

        current_times = self.lognormal_timestep_distribution(images.shape[0], sigmas, device=images.device).long()
        # weights = self.improved_loss_weighting(sigmas)#to do
        next_times = current_times + 1

        current_noise_image = images + self.image_time_product(
            noise,
            current_times,
        )

        next_noise_image = images + self.image_time_product(
            noise,
            next_times,
        )

        with torch.no_grad():
            target = self(
                current_noise_image,
                current_times,
            )

        loss = self.loss_fn(self(next_noise_image, next_times), target)

        self._loss_tracker(loss)
        self.log(
            "loss",
            self._loss_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        self._bins_tracker(_bins)
        self.log(
            "bins",
            self._bins_tracker,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        return self.optimizer_type(self.parameters(), lr=self.learning_rate)             

    def optimizer_step(self, *args, **kwargs) -> None:
        super().optimizer_step(*args, **kwargs)


    @property
    def bins(self) -> int:
        total_training_steps_prime = math.floor(
            self.trainer.estimated_stepping_batches
            /(math.log2(math.floor(self.bins_max / self.bins_min)) + 1)
            )
        num_timesteps = self.bins_min * math.pow(
            2, math.floor(self.trainer.global_step / total_training_steps_prime)
        )
        return min(num_timesteps, self.bins_max) + 1
    
    def karras_schedule(self, bins, device: torch.device = None):
        rho_inv = 1.0 / self.bins_rho
        steps = torch.arange(bins, device=device) / max(bins - 1, 1)
        sigmas = self.time_min**rho_inv + steps * (
            self.time_max**rho_inv - self.time_min**rho_inv
        )
        sigmas = sigmas**self.bins_rho

        return sigmas
    

    def lognormal_timestep_distribution(
        self,
        num_samples: int,
        sigmas,
        device = None
    ):

        pdf = torch.erf((torch.log(sigmas[1:]) - self.P_mean) / (self.P_std * math.sqrt(2))) - torch.erf((torch.log(sigmas[:-1]) - self.P_mean) / (self.P_std * math.sqrt(2)))
        pdf = pdf / pdf.sum()

        timesteps = torch.multinomial(pdf, num_samples, replacement=True).to(device)
        return timesteps


    def improved_loss_weighting(self, sigmas):
        return 1 / (sigmas[1:] - sigmas[:-1])
    


    @staticmethod
    def image_time_product(images: torch.Tensor, times: torch.Tensor):
        return torch.einsum("b c h w, b -> b c h w", images, times)
    
    # @rank_zero_only
    # def on_train_start(self) -> None:
    #     self.save_samples(
    #         f"{0:05}",
    #         num_samples=self.num_samples,
    #         steps=self.sample_steps,
    #         generator=torch.Generator(device=self.device).manual_seed(self.sample_seed),
    #     )

    # @rank_zero_only
    # def on_train_epoch_end(self) -> None:
    #     if (
    #         (self.trainer.current_epoch < 30)
    #         or ((self.trainer.current_epoch + 1) % self.save_samples_every_n_epoch == 0)
    #         or self.trainer.current_epoch == (self.trainer.max_epochs - 1)
    #     ):
    #         self.save_samples(
    #             f"{(self.current_epoch+1):05}",
    #             num_samples=self.num_samples,
    #             steps=self.sample_steps,
    #             generator=torch.Generator(device=self.device).manual_seed(
    #                 self.sample_seed
    #             ),
    #         )

    # @torch.no_grad()
    # def sample(
    #     self,
    #     num_samples: int = 16,
    #     steps: int = 1,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    # ) -> torch.Tensor:
    #     shape = (num_samples, self.channels, self.image_size, self.image_size)

    #     time = torch.tensor([self.time_max], device=self.device)

    #     images: torch.Tensor = self._forward(
    #         self.model,
    #         randn_tensor(shape, generator=generator, device=self.device) * time,
    #         time,
    #     )

    #     if steps <= 1:
    #         return images

    #     _timesteps = list(
    #         reversed(range(0, self.bins_max, self.bins_max // steps - 1))
    #     )[1:]
    #     _timesteps = [t + self.bins_max // ((steps - 1) * 2) for t in _timesteps]

    #     times = self.timesteps_to_times(
    #         torch.tensor(_timesteps, device=self.device), bins=150
    #     )

    #     for time in times:
    #         noise = randn_tensor(shape, generator=generator, device=self.device)
    #         images = images + math.sqrt(time.item() ** 2 - self.time_min**2) * noise
    #         images = self._forward(
    #             self.model,
    #             images,
    #             time[None],
    #         )

    #     return images

    # @torch.no_grad()
    # def save_samples(
    #     self,
    #     filename: str,
    #     num_samples: int = 16,
    #     steps: int = 1,
    #     generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    # ):
    #     samples = self.sample(
    #         num_samples=num_samples,
    #         steps=steps,
    #         generator=generator,
    #     )
    #     samples.mul_(0.5).add_(0.5)
    #     grid = make_grid(
    #         samples,
    #         nrow=math.ceil(math.sqrt(samples.size(0))),
    #         padding=self.image_size // 16,
    #     )

    #     save_image(
    #         grid,
    #         f"{self.samples_path}/{filename}.png",
    #         "png",
    #     )

    #     if isinstance(self.trainer.logger, WandbLogger):
    #         wandb.log(
    #             {
    #                 "samples": wandb.Image(to_pil_image(grid)),
    #             },
    #             commit=False,
    #             step=self.trainer.global_step,
    #         )

    #     del samples
    #     del grid
    #     torch.cuda.empty_cache()


    