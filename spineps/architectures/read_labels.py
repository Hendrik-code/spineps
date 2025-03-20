from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

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


class VertExactClass(Enum):
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
    T13 = 19
    L1 = 20
    L2 = 21
    L3 = 22
    L4 = 23
    L5 = 24
    L6 = 25
    # S = 24


class VertT13(Enum):
    Normal = 0
    T13 = 1


class VertGroup(Enum):
    C12 = 0
    C345 = 1
    C67 = 2
    T12 = 3
    T34 = 4
    T567 = 5
    T89 = 6
    T1011 = 7
    T123 = 8
    L12 = 9
    L34 = 10
    L56 = 11


def vert_label_to_vertrel(
    vertlabel: int,
    last_bwk,
    last_lwk,
    last_hwk=7,
    first_bwk=8,
    first_lwk=20,
) -> VertRel:
    last_hwk = 7
    first_bwk = 8
    first_lwk = 20

    l = VertRel.NOTHING
    if vertlabel == last_hwk:
        l = VertRel.LAST_HWK
    elif vertlabel == first_bwk:
        l = VertRel.FIRST_BWK
    elif last_bwk is not None and vertlabel == last_bwk:
        l = VertRel.LAST_BWK
    elif vertlabel == first_lwk:
        l = VertRel.FIRST_LWK
    elif last_lwk is not None and vertlabel == last_lwk:
        l = VertRel.LAST_LWK
    return l


def vert_class_to_region(vert_exact: VertExact) -> VertRegion:
    return VertRegion.HWS if vert_exact.value < 7 else VertRegion.BWS if 7 <= vert_exact.value < 19 else VertRegion.LWS


def vert_label_to_class(vertlabel: int) -> VertExact:
    return VertExact.T12 if vertlabel == 28 else VertExact(min(23, vertlabel - 1))


def vert_label_to_exactclass(vertlabel: int) -> VertExactClass:
    return (
        VertExactClass.T13
        if vertlabel == 28
        else VertExactClass(min(24, vertlabel - 1))
        if vertlabel <= 19
        else VertExactClass(min(25, vertlabel))
    )


vert_exact_to_group_dict: dict[VertExact, VertGroup] = {
    VertExact.C1: VertGroup.C12,
    VertExact.C2: VertGroup.C12,
    VertExact.C3: VertGroup.C345,
    VertExact.C4: VertGroup.C345,
    VertExact.C5: VertGroup.C345,
    VertExact.C6: VertGroup.C67,
    VertExact.C7: VertGroup.C67,
    VertExact.T1: VertGroup.T12,
    VertExact.T2: VertGroup.T12,
    VertExact.T3: VertGroup.T34,
    VertExact.T4: VertGroup.T34,
    VertExact.T5: VertGroup.T567,
    VertExact.T6: VertGroup.T567,
    VertExact.T7: VertGroup.T567,
    VertExact.T8: VertGroup.T89,
    VertExact.T9: VertGroup.T89,
    VertExact.T10: VertGroup.T1011,
    VertExact.T11: VertGroup.T1011,
    VertExact.T12: VertGroup.T123,
    VertExact.L1: VertGroup.L12,
    VertExact.L2: VertGroup.L12,
    VertExact.L3: VertGroup.L34,
    VertExact.L4: VertGroup.L34,
    VertExact.L5: VertGroup.L56,
}
vert_group_to_exact_dict: dict[VertGroup, list[VertExact]] = {}
for k, v in vert_exact_to_group_dict.items():
    if v not in vert_group_to_exact_dict:
        vert_group_to_exact_dict[v] = [k]
    else:
        vert_group_to_exact_dict[v].append(k)

vert_group_idx_to_exact_idx_dict: dict[int, list[int]] = {i.value: [gg.value for gg in g] for i, g in vert_group_to_exact_dict.items()}


def vert_class_to_group(vert_exact: VertExact) -> VertGroup:
    return vert_exact_to_group_dict[vert_exact]


def vertgrp_sequence_to_class(vertgrp: list[VertGroup]) -> list[VertExact]:
    # input must be sorted from top to bottom already!
    vert_exact_list: list[VertExact] = [None] * len(vertgrp)  # type: ignore

    for vg, vel in vert_group_to_exact_dict.items():
        if vg not in vertgrp:
            continue
        vertgrp_count = vertgrp.count(vg)
        vertgrp_idx = [i for i, val in enumerate(vertgrp) if val == vg]
        # all vertgrp instances there, trivial resolution
        if vertgrp_count == len(vel):
            for ii, i in enumerate(vertgrp_idx):
                vert_exact_list[i] = vel[ii]
        # only partial there, the group before or after determines exactness
        else:
            idx_before = min(vertgrp_idx) - 1
            idx_after = max(vertgrp_idx) + 1
            assert not (idx_before >= 0 and idx_after < len(vertgrp)), "partial grp sequence not possible"
            if idx_before >= 0:
                for ii, i in enumerate(vertgrp_idx):
                    vert_exact_list[i] = vel[ii]
            elif idx_after <= len(vertgrp) - 1:
                for ii, i in enumerate(vertgrp_idx[::-1]):
                    vert_exact_list[i] = vel[-(ii + 1)]
                # vert_exact_list[idx_before] = vel[0]
                # vert_exact_list[idx_after] = vel[-1]
    return vert_exact_list


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
    VERTEX = EnumLabelType, VertExactClass, "vert_exact2"  # exakt WK
    VT13 = EnumLabelType, VertT13, "vert_t13"  # exakt WK
    VERTREL = EnumLabelType, VertRel, "vert_rel"  # relative label (normal, last LWK, first BWK, ...)
    VERTGRP = EnumLabelType, VertGroup, "vert_group"  # exakt WK
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
    def group_2_n_channel(self) -> dict[str, int]:
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
@dataclass
class SubjectInfo:
    subject_name: int
    has_anomaly_entry: bool
    anomaly_entry: dict
    deleted_label: list[int]
    labelmap: dict
    is_remove: bool
    actual_labels: list[int]
    last_lwk: int
    last_bwk: int
    last_hwk: int = 7
    first_bwk: int = 8
    first_lwk: int = 20
    double_entries: list[int] = field(default_factory=list)

    @property
    def block(self) -> int:
        return int(str(self.subject_name)[:3])


# Get labels
def get_subject_info(
    subject_name: str | int,
    anomaly_dict: dict,
    vert_subfolders_int: list[int],
    anomaly_factor_condition: int = 0,
):
    double_entries = []
    labelmap = {}
    has_anomaly_entry = False
    anomaly_entry = {}
    deleted_label = []
    is_remove = False
    if int(subject_name) in anomaly_dict:
        anomaly_entry = anomaly_dict[subject_name]
        has_anomaly_entry = True
        if anomaly_entry["DeleteLabel"] is not None:
            deleted_label = [anomaly_entry["DeleteLabel"]]
        if bool(anomaly_entry["Remove"]):
            is_remove = True

        if bool(anomaly_entry["T11"]):
            labelmap = {i: i + 1 for i in range(19, 26)}
            double_entries = [17, 18, 20, 21]
        elif bool(anomaly_entry["T13"]):
            labelmap = {20: 28, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24}
            double_entries = [19, 28, 20, 21]
        elif anomaly_factor_condition == 0:
            double_entries = [18, 19, 20, 21]

    actual_labels = [labelmap.get(v, v) for v in vert_subfolders_int]
    #
    # last_hwk = 7
    # first_bwk = 8
    last_bwk = max([v for v in actual_labels if 7 < v <= 19 or v == 28]) if max(actual_labels) >= 18 else None
    # first_lwk = 20
    last_lwk = max([v for v in actual_labels if 22 < v < 26]) if max(actual_labels) >= 23 else None
    return SubjectInfo(
        subject_name=int(subject_name),
        has_anomaly_entry=has_anomaly_entry,
        anomaly_entry=anomaly_entry,
        actual_labels=actual_labels,
        deleted_label=deleted_label,
        is_remove=is_remove,
        labelmap=labelmap,
        last_lwk=last_lwk,
        last_bwk=last_bwk,
        double_entries=double_entries,
    )


def get_vert_entry(v: int, subject_info: SubjectInfo) -> tuple[int, dict]:
    entry: dict = {}

    v_actual = subject_info.labelmap.get(v, v)
    entry["subject_name"] = subject_info.subject_name
    entry["vert_rel"] = vert_label_to_vertrel(
        v_actual,
        subject_info.last_bwk,
        subject_info.last_lwk,
        last_hwk=subject_info.last_hwk,
        first_bwk=subject_info.first_bwk,
        first_lwk=subject_info.first_lwk,
    )
    entry["vert_cut"] = False
    entry["vert_exact"] = vert_label_to_class(v_actual)
    entry["vert_exact2"] = vert_label_to_exactclass(v_actual)
    entry["vert_group"] = vert_class_to_group(entry["vert_exact"])
    entry["vert_region"] = vert_class_to_region(entry["vert_exact"])
    entry["vert_t13"] = VertT13.T13 if v_actual == 28 else VertT13.Normal
    return v_actual, entry
