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

import nnunetv2.experiment_planning.plan_and_preprocess_api as pp
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, load_json, maybe_mkdir_p, write_pickle
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_raw
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager


def preprocess(
    dataset_ids: list[int],
    plans_identifier: str = "nnUNetPlans",
    configurations: tuple[str] | list[str] = ("2d", "3d_fullres", "3d_lowres"),  # type: ignore
    num_processes: int | tuple[int, ...] | list[int] = (8, 4, 8),
    compress=True,
    verbose: bool = False,
):
    for d in dataset_ids:
        preprocess_dataset(d, plans_identifier, configurations, num_processes, compress, verbose)


def preprocess_dataset(
    dataset_id: int,
    plans_identifier: str = "nnUNetPlans",
    configurations: tuple[str] | list[str] = ("2d", "3d_fullres", "3d_lowres"),  # type: ignore
    num_processes: int | tuple[int, ...] | list[int] = (8, 4, 8),
    compress=True,
    verbose: bool = False,
) -> None:
    if not isinstance(num_processes, list):
        num_processes = list(num_processes)  # type: ignore
    if len(num_processes) == 1:
        num_processes = num_processes * len(configurations)
    if len(num_processes) != len(configurations):
        raise RuntimeError(
            f"The list provided with num_processes must either have len 1 or as many elements as there are "
            f"configurations (see --help). Number of configurations: {len(configurations)}, length "
            f"of num_processes: "
            f"{len(num_processes)}"
        )

    dataset_name = pp.convert_id_to_dataset_name(dataset_id)
    print(f"Preprocessing dataset {dataset_name}")
    plans_file = join(nnUNet_preprocessed, dataset_name, plans_identifier + ".json")  # type: ignore
    plans_manager = PlansManager(plans_file)
    for n, c in zip(num_processes, configurations, strict=False):
        print(f"Configuration: {c}...")
        if c not in plans_manager.available_configurations:
            print(f"INFO: Configuration {c} not found in plans file {plans_identifier + '.json'} of " f"dataset {dataset_name}. Skipping.")
            continue
        preprocessor = FastPreprocessor(verbose=verbose, compress=compress)
        preprocessor.run(dataset_id, c, plans_identifier, num_processes=n)

    # copy the gt to a folder in the nnUNet_preprocessed so that we can do validation even if the raw data is no
    # longer there (useful for compute cluster where only the preprocessed data is available)
    from distutils.file_util import copy_file

    maybe_mkdir_p(join(nnUNet_preprocessed, dataset_name, "gt_segmentations"))  # type: ignore
    dataset_json = load_json(join(nnUNet_raw, dataset_name, "dataset.json"))  # type: ignore
    dataset = pp.get_filenames_of_train_images_and_targets(join(nnUNet_raw, dataset_name), dataset_json)  # type: ignore
    # only copy files that are newer than the ones already present
    for k in dataset:
        copy_file(
            dataset[k]["label"],
            join(nnUNet_preprocessed, dataset_name, "gt_segmentations", k + dataset_json["file_ending"]),  # type: ignore
            update=True,  # type: ignore
        )  # type: ignore


class FastPreprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = True, compress=True):
        super().__init__(verbose)
        self.compress = compress

    def run_case_save(
        self,
        output_filename_truncated: str,
        image_files: list[str],
        seg_file: str,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict | str,
    ):
        data, seg, properties = self.run_case(image_files, seg_file, plans_manager, configuration_manager, dataset_json)
        # print('dtypes', data.dtype, seg.dtype)
        # print(data.dtype, data.shape, data.max(), data.min())
        if self.compress:
            np.savez_compressed(output_filename_truncated + ".npz", data=data.astype(np.float16), seg=seg)
        else:
            np.savez(output_filename_truncated + ".npz", data=data.astype(np.float16), seg=seg)
        write_pickle(properties, output_filename_truncated + ".pkl")
