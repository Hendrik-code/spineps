"""Enumerations describing image modalities, acquisitions, model types and pipeline phases used across SPINEPS."""

from __future__ import annotations

from enum import Enum, EnumMeta, auto

from typing_extensions import Self


class MetaEnum(EnumMeta):
    """Enum metaclass enabling ``item in EnumClass`` membership tests by member name."""

    def __contains__(cls, item):
        """Return whether ``item`` names a member of the enum.

        Args:
            item: Candidate member name to test for membership.

        Returns:
            bool: True if ``item`` is a valid member name of the enum, False otherwise.
        """
        try:
            cls[item]
        except ValueError:
            return False
        return True


class Enum_Compare(Enum, metaclass=MetaEnum):
    """Base enum that compares equal to other enums by name/value and to plain strings by name.

    Provides string-friendly equality, hashing and representation so members can be compared
    against and interchanged with their string names throughout the pipeline.
    """

    def __eq__(self, __value: object) -> bool:  # noqa: PYI063
        """Compare this member against another enum or a string.

        Args:
            __value (object): Another enum member or a string holding a member name.

        Returns:
            bool: True if the other enum matches by name and value, or the string matches this member's name.
        """
        if isinstance(__value, Enum):
            return self.name == __value.name and self.value == __value.value
        elif isinstance(__value, str):
            return self.name == __value
        else:
            return False

    def __str__(self) -> str:
        """Return the member as ``ClassName.MEMBER``.

        Returns:
            str: Human-readable identifier of the member.
        """
        return f"{type(self).__name__}.{self.name}"

    def __repr__(self) -> str:
        """Return the same string as :meth:`__str__`.

        Returns:
            str: Human-readable identifier of the member.
        """
        return str(self)

    def __hash__(self) -> int:
        """Return the member's integer value as its hash.

        Returns:
            int: The member's value, used for hashing.
        """
        return self.value


class Modality(Enum_Compare):
    """Image modality of an input scan.

    Members cover the MRI sequences and other image types SPINEPS can handle, e.g. T2-weighted (T2w),
    T1-weighted (T1w), Vibe/Dixon, CT, an existing segmentation (SEG), multi-planar reconstruction (MPR),
    proton density (PD) and FLAIR.
    """

    T2w = auto()
    T1w = auto()
    Vibe = auto()
    CT = auto()
    SEG = auto()
    MPR = auto()
    PD = auto()
    FLAIR = auto()

    @classmethod
    def format_keys(cls, modalities: Self | list[Self]) -> list[str]:
        """Map modality members to the BIDS/file-name string keys that denote them.

        Args:
            modalities (Self | list[Self]): A single modality member or a list of them.

        Returns:
            list[str]: All file-name/format keys associated with the given modalities.

        Raises:
            NotImplementedError: If a modality has no associated keys defined.
        """
        if not isinstance(modalities, list):
            modalities = [modalities]
        result = []
        for modality in modalities:
            if modality == Modality.CT:
                result += ["CT", "ct"]
            elif modality == Modality.SEG:
                result += ["msk", "seg"]
            elif modality == Modality.T1w:
                result += ["T1w", "t1", "T1", "T1c"]
            elif modality == Modality.T2w:
                result += ["T2w", "dixon", "mr", "t2", "T2", "T2c"]
            elif modality == Modality.Vibe:
                result += ["t1dixon", "vibe", "mevibe", "GRE"]
            elif modality == Modality.MPR:
                result += ["mpr", "MPR", "Mpr"]
            else:
                raise NotImplementedError(modality)
        return result


class Acquisition(Enum_Compare):
    """Acquisition plane of a scan.

    Members denote the imaging plane: ``sag`` (sagittal), ``cor`` (coronal), ``ax`` (axial)
    and ``iso`` (isotropic / no dominant plane).
    """

    sag = auto()
    cor = auto()
    ax = auto()
    iso = auto()

    @classmethod
    def format_keys(cls, acquisition: Self) -> list[str]:
        """Map an acquisition member to the file-name string keys that denote it.

        Args:
            acquisition (Self): The acquisition plane member.

        Returns:
            list[str]: All file-name/format keys associated with the given acquisition.

        Raises:
            NotImplementedError: If the acquisition has no associated keys defined.
        """
        if acquisition == Acquisition.ax:
            return ["axial", "ax", "axl"]
        elif acquisition == Acquisition.cor:
            return ["coronal", "cor"]
        elif acquisition == Acquisition.sag:
            return ["sagittal", "sag"]
        elif acquisition == Acquisition.iso:
            return ["iso", "ISO"]
        else:
            raise NotImplementedError(acquisition)


class SpinepsPhase(Enum_Compare):
    """Stage of the SPINEPS pipeline: semantic segmentation, vertebra instance segmentation or labeling."""

    SEMANTIC = auto()
    INSTANCE = auto()
    LABELING = auto()


class ModelType(Enum_Compare):
    """Kind of model backing an inference config: an nnU-Net, a plain U-Net or a classifier."""

    nnunet = auto()
    unet = auto()
    classifier = auto()


class InputType(Enum_Compare):
    """Type of input channel fed to a model.

    ``img`` is the default image input and ``seg`` a segmentation input. The remaining members
    are the Dixon/Vibe channels: in-phase (``ip``), out-of-phase (``oop``), ``water`` and ``fat``.
    """

    img = auto()  # default: image input
    seg = auto()  # segmentation input
    # For Vibe
    ip = auto()  # inphase
    oop = auto()  # out of phase
    water = auto()  # water
    fat = auto()  # fat


class OutputType(Enum_Compare):
    """Type of model output: a segmentation (``seg``), softmax logits or an uncertainty map (``unc``)."""

    seg = auto()
    # seg_modelres = auto()
    softmax_logits = auto()
    unc = auto()


class ErrCode(Enum_Compare):
    """Status/error codes returned by pipeline steps.

    Indicates success (``OK``), that outputs already exist (``ALL_DONE``), a model/input compatibility
    problem (``COMPATIBILITY``), an unknown failure (``UNKNOWN``), an empty mask or input (``EMPTY``)
    or mismatched shapes (``SHAPE``).
    """

    OK = auto()
    ALL_DONE = auto()  # outputs are already there
    COMPATIBILITY = auto()  # compatibility issue between model and input
    UNKNOWN = auto()  # unknown issue
    EMPTY = auto()  # issue that the mask or input is empty
    SHAPE = auto()  # issue that shapes do not match


if __name__ == "__main__":
    mod = Modality.T2w
    aq = Acquisition.sag
    print(mod)
    print(aq)
