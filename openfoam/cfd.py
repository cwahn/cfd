from __future__ import annotations

from abc import ABCMeta
from enum import Enum
from pathlib import Path
from termios import VT1
from typing import Generic, List, Literal, NewType, Set, TypeVar, TypeVarTuple, Optional
from attr import dataclass
from prep.src import Vector

# Fix the name of bounding patchs to xMin, xMax, yMin, yMax, zMin, zMax
# First define patch names
# And define solver type, with turbulence model -> Will define required fields
#


Scalar = float

Vector = tuple[Scalar, Scalar, Scalar]

Tensor = tuple[Vector, Vector, Vector]

DataE = str | Scalar | Vector | Tensor | Path


@dataclass
class KeywordE:
    keyword: str
    data: List[DataE] | tuple[DataE] # ? Free, and fully specified are good. What about essential and optioanl?


@dataclass
class Dictioanry:
    keyword: str
    entries: List[KeywordE] | tuple[KeywordE]


# class DataFileVersion(Enum):
#     V1 = 0
#     V2 = 1

@dataclass
class DataFileHeaderVersion(KeywordE):
    keyword: Literal["version"]
    data: tuple[Literal["V1", "V2"]] # List with one elem

# class DataFileFormat(Enum):
#     ASCII = 0
#     Binary = 1

@dataclass
class DataFileHeaderFormat(KeywordE):
    keyword: Literal["format"]
    data: tuple[Literal["ASCII", "Binary"]]

@dataclass
class DataFileHeaderLocation(KeywordE):
    keyword: Literal["location"]
    data: tuple[Path]


# class OpenFoamClass(Enum):
#     Dictioanry = 0
#     # ...

@dataclass
class DataFileHeaderClass(KeywordE):
    keyword: Literal["class"]
    data: tuple[str]

@dataclass
class DataFileHeaderObject(KeywordE):
    keyword: Literal["object"]
    data: tuple[str]


@dataclass
class DataFileHeader(Dictioanry):
    keyword : Literal["FoamFile"]
    entries: tuple[
        DataFileHeaderVersion,
        DataFileHeaderFormat,
        Optional[DataFileHeaderLocation],
        DataFileHeaderClass,
        DataFileHeaderObject # Also a file name
    ]
    # version: DataFileVersion
    # format: DataFileFormat
    # location: Optional[Path]
    # openfoam_class: OpenFoamClass
    # object: OpenFoamObject  # Also a file name

@dataclass
class DataFile:
    header: DataFileHeader
    body:  Set[Dictioanry]