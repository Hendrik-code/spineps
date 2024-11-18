from abc import ABC, abstractmethod
from enum import Enum

import numpy as np

TRUE_KEYS = [True, "True", "true", 1, "wahr", "Wahr", "Ja", "ja", "-", "1?"]
FALSE_KEYS = [False, "False", "false", 0, "falsch", "Falsch", "Nein", "nein"]


class VertRegion(Enum):
    HWS = 0
    BWS = 1
    LWS = 2


class VertRel(Enum):
    NOTHING = 0
    LAST_HWK = 1
    #
    FIRST_BWK = 2
    LAST_BWK = 3
    #
    FIRST_LWK = 4
    LAST_LWK = 5


class VertExact(Enum):
    C1 = 0
    C2 = 1
    C3 = 2
    C4 = 3
    C5 = 4
    C6 = 5
    C7 = 6
    T1 = 7
    T2 = 8
    T3 = 9
    T4 = 10
    T5 = 11
    T6 = 12
    T7 = 13
    T8 = 14
    T9 = 15
    T10 = 16
    T11 = 17
    T12 = 18
    # T13 = 18
    L1 = 19
    L2 = 20
    L3 = 21
    L4 = 22
    L5 = 23
    # L6 = 23
    # S = 24


class LabelType(ABC):
    def __init__(self, column_name: str | list[str], *args, **kwargs) -> None:  # noqa: ARG002
        if not isinstance(column_name, list):
            column_name = [column_name]
        self.column_name = column_name

    def __call__(self, entry_dict: dict):
        entry = self.get_entry(entry_dict)
        return self.convert_to_label(entry)

    def get_entry(self, entry: dict) -> str | int | list[str | int]:
        entries = [entry[c] for c in self.column_name]
        if len(entries) == 1:
            return entries[0]
        return entries

    @property
    @abstractmethod
    def number_of_channel(self) -> str | int | list[str | int]:
        pass

    @abstractmethod
    def convert_to_label(self, entry: str):
        pass


class EnumLabelType(LabelType):
    def __init__(self, enum: Enum, column_name: str, *args, **kwargs) -> None:  # noqa: ARG002
        super().__init__(column_name)
        self.enum = enum
        self.n_channel = len(enum)

    @property
    def number_of_channel(self) -> int:
        return self.n_channel

    def convert_to_label(self, entry: Enum):
        label = list(np.zeros(self.number_of_channel, dtype=int))
        idx = entry.value
        label[idx] = 1
        return label


class BinaryLabelTypeDummy(LabelType):
    def __init__(self, column_name: str | list[str], *args, **kwargs) -> None:
        super().__init__(column_name, *args, **kwargs)

    @property
    def number_of_channel(self) -> int:
        return 2

    def convert_to_label(self, entry: str | int) -> int:
        assert not isinstance(entry, list), entry
        if entry in TRUE_KEYS:
            return [1, 0]
        elif entry in FALSE_KEYS:
            return [0, 1]
        raise AssertionError(f"entry {entry} not defined as BinaryLabel")


class Target(Enum):
    REGION = EnumLabelType, VertRegion, "vert_region"  # HWS, BWS, LWS
    VERT = EnumLabelType, VertExact, "vert_exact"  # exakt WK
    VERTREL = EnumLabelType, VertRel, "vert_rel"  # relative label (normal, last LWK, first BWK, ...)
    # for each above is alone multiclass, so softmax afterwards target-wise
    #
    FULLYVISIBLE = BinaryLabelTypeDummy, "vert_cut", "vert_cut"


TARGET_COLUMN_OPTIONS = [s.value[1] for s in Target]


class Objectives:
    def __init__(
        self,
        objectives: list[Target],
        as_group: bool = True,
    ) -> None:
        self.__as_group = as_group
        self.targets: list[Target] = objectives
        self.__objective_labels: list[LabelType] = []
        #
        for o in objectives:
            # Horizontal_Flip_Dict
            not_flipped_target = o.value[0](o.value[1], o.value[2])
            self.__objective_labels.append(not_flipped_target)

        self.__n_channel_p_group = [o.number_of_channel for o in self.__objective_labels]
        self.__n_channel = sum(self.__n_channel_p_group)
        self.__required_dict_keys = list(set(flatten([o.value[2] for o in objectives])))

    @property
    def n_channel_p_group(self):
        return self.__n_channel_p_group

    @property
    def n_channel(self):
        return self.__n_channel

    @property
    def group_2_n_channel(self):
        return {self.targets[idx].name: self.n_channel_p_group[idx] for idx in range(len(self.targets))}

    @property
    def required_dict_keys(self):
        return self.__required_dict_keys

    def __call__(
        self,
        entry: dict,
    ) -> list[int]:
        entry_keys = entry.keys()
        for r in self.required_dict_keys:
            assert r in entry_keys, f"required {r} not in entry_keys, got {entry_keys}"

        #
        labels = []
        labels_grouped = []
        try:
            list_of_ordered_objectives = self.__objective_labels

            for labeltype in list_of_ordered_objectives:
                labeladd = labeltype(entry)
                if not isinstance(labeladd, list):
                    labeladd = [labeladd]
                labels += labeladd
                labels_grouped.append(labeladd)
        except AssertionError:  # nan binary label
            labels = None
        except AttributeError:  # nan Pathology label
            labels = None
        return labels if not self.__as_group else {self.targets[idx].name: labels_grouped[idx] for idx in range(len(self.targets))}


def flatten(a: list[str | int | list[str] | list[int]]):
    # a = itertools.chain(*a)
    if isinstance(a, (str, int)):
        yield a
    else:
        for b in a:
            yield from flatten(b)


###

# Eval-pipeline zuerst
# sensitivity, recall, AUC, ROC, F1, MCC
# dann MONAI baseline bauen mit Resnet, Densenet, ViT
if __name__ == "__main__":
    objectives = Objectives(
        [
            Target.FULLYVISIBLE,
            Target.REGION,
            Target.VERTREL,
            Target.VERT,
        ],
        as_group=True,
    )

    entry_dict = {
        "vert_exact": VertExact.L1,
        "vert_region": VertRegion.LWS,
        "vert_rel": VertRel.FIRST_LWK,
        "vert_cut": True,
    }

    label = objectives(entry_dict)
    print(label)
