from enum import Enum, auto

from typing_extensions import Self


class Enum_Compare(Enum):
    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, Enum):
            return self.name == __value.name and self.value == __value.value
        elif isinstance(__value, str):
            return self.name == __value
        else:
            return False

    def __str__(self) -> str:
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self.value


class Modality(Enum_Compare):
    """Describes image modality

    Args:
        Enum_Compare (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    T2w = auto()
    T1w = auto()
    Vibe = auto()
    CT = auto()
    SEG = auto()

    @classmethod
    def format_keys(cls, modalities: Self | list[Self]) -> list[str]:
        if not isinstance(modalities, list):
            modalities = [modalities]
        result = []
        for modality in modalities:
            if modality == Modality.CT:
                result += ["CT", "ct"]
            elif modality == Modality.SEG:
                result += ["msk", "seg"]
            elif modality == Modality.T1w:
                result += ["T1w", "t1", "T1"]
            elif modality == Modality.T2w:
                result += ["T2w", "dixon", "mr", "t2", "T2"]
            elif modality == Modality.Vibe:
                result += ["t1dixon", "vibe"]
            else:
                raise NotImplementedError(modality)
        return result


class Acquisition(Enum_Compare):
    """Describes Acquisition (sag = sagittal, cor = coronal, ax = axial)

    Args:
        Enum_Compare (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        _type_: _description_
    """

    sag = auto()
    cor = auto()
    ax = auto()

    @classmethod
    def format_keys(cls, acquisition: Self) -> list[str]:
        if acquisition == Acquisition.ax:
            return ["axial", "ax", "axl"]
        elif acquisition == Acquisition.cor:
            return ["coronal", "cor"]
        elif acquisition == Acquisition.sag:
            return ["sagittal", "sag"]
        else:
            raise NotImplementedError(acquisition)


class ModelType(Enum_Compare):
    nnunet = auto()
    unet = auto()


class InputType(Enum_Compare):
    img = auto()  # default: image input
    seg = auto()  # segmentation input
    # For Vibe
    ip = auto()  # inphase
    oop = auto()  # out of phase
    water = auto()  # water
    fat = auto()  # fat


class OutputType(Enum_Compare):
    seg = auto()
    # seg_modelres = auto()
    softmax_logits = auto()
    unc = auto()


class ErrCode(Enum_Compare):
    OK = auto()
    ALL_DONE = auto()  # outputs are already there
    COMPATIBILITY = auto()  # compatibilty issue between model and input
    UNKNOWN = auto()  # unknown issue
    EMPTY = auto()  # issue that the mask or input is empty
    SHAPE = auto()  # issue that shapes do not match


if __name__ == "__main__":
    mod = Modality.T2w
    aq = Acquisition.sag
    print(mod)
    print(aq)
