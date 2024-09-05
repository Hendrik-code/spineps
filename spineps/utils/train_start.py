#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from functools import partial
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from time import sleep, time

import nnunetv2.experiment_planning.plan_and_preprocess_api as pp
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.run.run_training import get_trainer_from_args, maybe_load_checkpoint
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from torch.backends import cudnn

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
#### Settings ####
out_base = Path(__file__).parent.parent / "nnUNet/"

plans = "nnUNetPlannerResEncL"
verify_dataset_integrity = True

idx = 94
name = None
debug = False
gpus = ["1"]
num_epochs = 250
num_iterations_per_epoch = 1000  # 250 # I use 1/4 epochs but 4 times the iteration to save eval time.
num_folds = 1
single_gpu = len(gpus) == 1
patch_size: tuple[int, ...] | None = None
batch_size: int | None = None
batch_size_val = 1  # With a large amount of channels we run out of memory
overwrite_target_spacing = None
# TODO Check if gpu exist
####################


@dataclass
class plan_and_preprocess_args:
    d: list[int] = field(default_factory=lambda: [idx])
    fpe: str = "DatasetFingerprintExtractor"
    npfp: int = 8
    verify_dataset_integrity = verify_dataset_integrity
    no_pp: bool = False
    clean: bool = False
    pl: str = "ExperimentPlanner"
    gpu_memory_target: int | None = None
    preprocessor_name: str = "DefaultPreprocessor"
    overwrite_target_spacing: list[float] | None = None
    overwrite_plans_name: str = plans
    c: list[str] = field(default_factory=lambda: ["3d_fullres"])  # default=["2d", "3d_fullres", "3d_lowres"],
    np: list[int] = field(default_factory=lambda: [16])  # [8, 4, 8]
    verbose: bool = False


def run_nn_unet(fold=0):
    print("Start fold", fold)

    os.environ["nnUNet_raw"] = str(out_base / "nnUNet_raw")  # noqa: SIM112
    os.environ["nnUNet_preprocessed"] = str(out_base / "nnUNet_preprocessed")  # noqa: SIM112
    os.environ["nnUNet_results"] = str(out_base / "nnUNet_results")  # noqa: SIM112
    if torch.cuda.device_count() > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(fold % (torch.cuda.device_count() - 1) + 1)
    best_checkpoints = list(Path(out_base / "nnUNet_results").glob(f"Dataset{idx:03}*/*_3d_full*/fold_{fold}/checkpoint_best.pth"))
    print("existing Trainigs: ", best_checkpoints)
    run_training(
        str(idx),
        "3d_fullres",
        str(fold),
        "nnUNetTrainer" if turn_on_mirroring else "nnUNetTrainerNoMirroring",
        plans,
        patch_size=patch_size,
        continue_training=len(best_checkpoints) != 0,
        num_epochs=num_epochs,
        num_iterations_per_epoch=num_iterations_per_epoch,
        save_every=1,
    )
    sleep(1)


_set_num_interop_threads = False


def run_training(
    dataset_name_or_id: str | int,
    configuration: str,
    fold: int | str,
    trainer_class_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    pretrained_weights: str | None = None,
    num_gpus: int = 1,
    use_compressed_data: bool = False,
    export_validation_probabilities: bool = False,
    continue_training: bool = False,
    only_run_validation: bool = False,
    disable_checkpointing: bool = False,
    val_with_best: bool = False,
    device_str: str = "cuda",
    initial_lr=1e-2,
    weight_decay=3e-5,
    oversample_foreground_percent=0.33,
    num_iterations_per_epoch=250,
    num_val_iterations_per_epoch=50,  # 50
    num_epochs=250,  # 1000
    current_epoch=0,
    enable_deep_supervision=True,
    save_every=5,  # 50
    patch_size: list[int] | None = None,
):
    global _set_num_interop_threads  # noqa: PLW0603
    if device_str == "cpu":
        # let's allow torch to use hella threads
        import multiprocessing

        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device("cpu")
    elif device_str == "cuda":
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        if not _set_num_interop_threads:
            torch.set_num_interop_threads(1)
            _set_num_interop_threads = True
        device = torch.device("cuda")
    else:
        device = torch.device("mps")
    if isinstance(fold, str) and fold != "all":
        try:
            fold = int(fold)
        except ValueError:
            print(f'Unable to convert given value for fold to int: {fold}. fold must bei either "all" or an integer!')
            raise

    if val_with_best:
        assert not disable_checkpointing, "--val_best is not compatible with --disable_checkpointing"

    if num_gpus > 1:
        raise NotImplementedError("DDPM")
    else:
        nnunet_trainer = get_trainer_from_args(
            dataset_name_or_id, configuration, fold, trainer_class_name, plans_identifier, use_compressed_data, device=device
        )
        nnunet_trainer.initial_lr = initial_lr
        nnunet_trainer.weight_decay = weight_decay
        nnunet_trainer.oversample_foreground_percent = oversample_foreground_percent
        nnunet_trainer.num_iterations_per_epoch = num_iterations_per_epoch
        nnunet_trainer.num_val_iterations_per_epoch = num_val_iterations_per_epoch
        nnunet_trainer.num_epochs = num_epochs
        nnunet_trainer.current_epoch = current_epoch
        nnunet_trainer.enable_deep_supervision = enable_deep_supervision  # type: ignore
        nnunet_trainer.save_every = save_every
        #### Highjack functions ####
        nnunet_trainer.get_plain_dataloaders = partial(get_plain_dataloaders_highjack, nnunet_trainer)
        #### Highjack functions END ###
        if patch_size is not None:
            nnunet_trainer.configuration_manager.configuration["patch_size"] = patch_size

        if disable_checkpointing:
            nnunet_trainer.disable_checkpointing = disable_checkpointing

        assert not (continue_training and only_run_validation), "Cannot set --c and --val flag at the same time. Dummy."
        maybe_load_checkpoint(nnunet_trainer, continue_training, only_run_validation, pretrained_weights)  # type: ignore

        if torch.cuda.is_available():
            cudnn.deterministic = False
            cudnn.benchmark = True

        if not only_run_validation:
            # nnunet_trainer.run_training()
            run_training_highjack(nnunet_trainer)

        if val_with_best:
            nnunet_trainer.load_checkpoint(join(nnunet_trainer.output_folder, "checkpoint_best.pth"))
        nnunet_trainer.perform_actual_validation(export_validation_probabilities)


def get_plain_dataloaders_highjack(self: nnUNetTrainer, initial_patch_size: tuple[int, ...], dim: int):
    dataset_tr, dataset_val = self.get_tr_and_val_datasets()

    if dim == 2:
        dl_tr = nnUNetDataLoader2D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,  # type: ignore
            pad_sides=None,  # type: ignore
        )
        dl_val = nnUNetDataLoader2D(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,  # type: ignore
            pad_sides=None,  # type: ignore
        )
    else:
        dl_tr = nnUNetDataLoader3D(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,  # type: ignore
            pad_sides=None,  # type: ignore
        )
        dl_val = nnUNetDataLoader3D(
            dataset_val,
            batch_size_val,  # self.batch_size // 2,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,  # type: ignore
            pad_sides=None,  # type: ignore
        )
    return dl_tr, dl_val


def run_training_highjack(self: nnUNetTrainer):
    self.on_train_start()

    for epoch in range(self.current_epoch, self.num_epochs):
        t = time()
        self.on_epoch_start()

        self.on_train_epoch_start()
        train_outputs = []

        for batch_id in range(self.num_iterations_per_epoch):
            x = time() - t
            print(
                f"{epoch}:{batch_id:04}/{self.num_iterations_per_epoch:04}",
                " time:",
                str(timedelta(seconds=x)),
                "ETA:",
                str(timedelta(seconds=x / (max(1, batch_id)) * self.num_iterations_per_epoch)),
                end="\r",
            )
            train_outputs.append(self.train_step(next(self.dataloader_train)))  # type: ignore
        print(f"{epoch}:{batch_id:05}/{self.num_iterations_per_epoch:05}", " time", str(timedelta(seconds=time() - t)))
        self.on_train_epoch_end(train_outputs)
        torch.cuda.empty_cache()
        t = time()
        with torch.no_grad():
            self.on_validation_epoch_start()
            val_outputs = []
            for batch_id in range(self.num_val_iterations_per_epoch):
                print(f"{batch_id:05}/{self.num_val_iterations_per_epoch:05}", " time", str(timedelta(seconds=time() - t)), end="\r")

                val_outputs.append(self.validation_step(next(self.dataloader_val)))  # type: ignore
            self.on_validation_epoch_end(val_outputs)

        self.on_epoch_end()

    self.on_train_end()


if __name__ == "__main__":
    os.environ["nnUNet_raw"] = str(out_base / "nnUNet_raw")  # noqa: SIM112
    os.environ["nnUNet_preprocessed"] = str(out_base / "nnUNet_preprocessed")  # noqa: SIM112
    os.environ["nnUNet_results"] = str(out_base / "nnUNet_results")  # noqa: SIM112
    ### LOAD form dataset ####
    dataset_folder_name = f"Dataset{idx:03}"
    if name is not None and name != "":
        dataset_folder_name += "_" + name
    try:
        n = next((out_base / "nnUNet_raw").glob(f"Dataset{idx:03}*")).name
        assert n == dataset_folder_name, f"this {idx=} already exist with an other name: {n}"
    except StopIteration:
        pass
    with open(out_base / "nnUNet_raw" / dataset_folder_name / "dataset.json") as f:
        ds_dic = json.load(f)
    patch_size = ds_dic.get("patch_size", patch_size)
    turn_on_mirroring = ds_dic.get("turn_on_mirroring", False)
    no_patch_window_override = ds_dic.get("no_patch_window_override", False)
    if no_patch_window_override:
        patch_size = None
    overwrite_target_spacing = ds_dic.get("spacing", overwrite_target_spacing)
    overwrite_target_spacing = [float(a) for a in overwrite_target_spacing] if overwrite_target_spacing is not None else None
    num_epochs = int(ceil(num_epochs))
    ############
    from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw, nnUNet_results

    print(nnUNet_results)
    print(nnUNet_preprocessed)
    print(nnUNet_raw)

    from spineps.utils.fastProcessor import preprocess

    if not (out_base / "nnUNet_preprocessed" / dataset_folder_name / (plans + ".json")).exists():
        args = plan_and_preprocess_args(overwrite_target_spacing=overwrite_target_spacing, pl=plans)
        print("Fingerprint extraction...")
        pp.extract_fingerprints(args.d, args.fpe, args.npfp, args.verify_dataset_integrity, args.clean, args.verbose)
        # experiment planning
        print("Experiment planning...")
        pp.plan_experiments(
            args.d, args.pl, args.gpu_memory_target, args.preprocessor_name, args.overwrite_target_spacing, args.overwrite_plans_name
        )
        # preprocessing
        if not args.no_pp:
            print("Preprocessing...")
            preprocess(args.d, args.overwrite_plans_name, args.c, args.np, args.verbose)
    pap = out_base / "nnUNet_preprocessed" / f"Dataset{idx:03}" / f"{plans}.json"
    with open(pap) as f:
        d = json.load(f)

    if patch_size is not None:
        d["configurations"]["3d_fullres"]["patch_size"] = patch_size
        with open(pap, "w") as f:
            json.dump(d, f, indent=3)
    if batch_size is not None:
        d["configurations"]["3d_fullres"]["batch_size"] = batch_size
        with open(pap, "w") as f:
            json.dump(d, f, indent=3)

    if len(sys.argv) != 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
        run_nn_unet(int(sys.argv[1]))
        sys.exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus[0]
    if single_gpu:
        [run_nn_unet(i) for i in range(num_folds)]
    else:
        print("MULTI PROCESSING")
        folds = list(reversed([str(i) for i in (range(len(gpus), 5))]))

        def call(i, gpu):
            print("start", i, "gpu", gpu)
            subprocess.check_call(["python", (Path(__file__).absolute()), str(i), str(gpu)])

        from functools import partial

        with Pool(len(gpus)) as p:

            def call_back(_, gpu=None):
                if len(folds) == 0:
                    return
                i = folds.pop()
                print("folds remaining", folds)
                r = p.apply_async(call, (i, gpu), callback=partial(call_back, gpu=gpu))
                results.append(r)

            results = [p.apply_async(call, (i, gpu), callback=partial(call_back, gpu=gpu)) for i, gpu in enumerate(gpus)]
            while True:
                if len(results) == 0:
                    time.sleep(10)
                    if len(results) == 0:
                        break
                results.pop().wait()

            p.close()
            p.join()
