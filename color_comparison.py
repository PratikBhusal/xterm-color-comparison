#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
from typing import (
    NamedTuple,
    ParamSpec,
    TypeVar,
    Any,
    Callable,
    Self,
    TypeAlias,
)
from functools import reduce, partial
import numpy
import numpy.typing as npt
from numpy import longdouble, double
from numpy._typing import _16Bit, _32Bit
from colour import (  # type: ignore[attr-defined]
    chromatic_adaptation,
    sRGB_to_XYZ,
    XYZ_to_sRGB,
    Oklab_to_XYZ,
    XYZ_to_Oklab,
    Oklab_to_Oklch,
    Oklch_to_Oklab,
    delta_E,
)
from colour.models import xy_to_XYZ
from colour.temperature import CCT_to_xy_CIE_D
import re
import json
import heapq
from pipe import select  # type: ignore[import]
import typer


P = ParamSpec("P")
T = TypeVar("T")
floating: TypeAlias = float | double | longdouble
# colour-science stubs declare return dtype as this union, though runtime is always float64
ColourFloat: TypeAlias = numpy.floating[_16Bit] | numpy.floating[_32Bit] | numpy.float64

dummy_text: str = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis a convallis ante,
vel volutpat risus. In vitae sapien a sapien ...
""".replace("\n", "")

OKLCH_PATTERN: re.Pattern[str] = re.compile(r"oklch\(([\d.]+)%\s+([\d.]+)\s+([\d.]+)\)")

app: typer.Typer = typer.Typer()


def compose_left_to_right(  # type: ignore[misc]
    first_function: Callable[P, Any],
    *other_functions: Callable[[Any], Any | T],
) -> Callable[P, T]:
    if len(other_functions) == 0:
        return first_function

    def compose2(  # type: ignore[misc]
        f: Callable[P, Any] | Callable[[Any], Any], g: Callable[[Any], Any | T]
    ) -> Callable[..., Any | T]:
        return lambda *args, **kwargs: g(f(*args, **kwargs))

    return reduce(compose2, other_functions, first_function)


class StandardRGB(NamedTuple):
    """
    Standard RGB Color Space
    """

    red: floating
    green: floating
    blue: floating

    # Ignore mypy decorated any on classmethod decorator
    @classmethod  # type: ignore[misc]
    def create(cls, hex_color: str) -> Self:
        hex_int: int = int(hex_color.removeprefix("#"), 16)

        return cls(
            longdouble((hex_int >> 16) & 255) / 255,
            longdouble((hex_int >> 8) & 255) / 255,
            longdouble(hex_int & 255) / 255,
        )


class Oklab(NamedTuple):
    lightness: floating
    green_red: floating
    blue_yellow: floating


class XtermColorDifference(NamedTuple):
    color_id: int
    name: str
    hex_string: str
    delta_e: floating


class TailwindColorDifference(NamedTuple):
    name: str
    shade: str
    value: str
    delta_e: floating


def rgb_colored_text(
    r: int, g: int, b: int, text: str, background: bool = False
) -> str:
    return f"\033[{48 if background else 38};2;{r};{g};{b}m{text}\033[0;0m"


def xterm_colored_text(color_id: int, text: str, background: bool = False) -> str:
    return f"\033[{48 if background else 38};5;{color_id}m{text}\033[0;0m"


def _to_oklab(arr: npt.NDArray[ColourFloat]) -> Oklab:
    return Oklab(arr[0], arr[1], arr[2])


hex_to_oklab: Callable[[str], Oklab] = compose_left_to_right(
    StandardRGB.create, sRGB_to_XYZ, XYZ_to_Oklab, _to_oklab
)


def _to_ndarray(oklab: Oklab) -> npt.NDArray[numpy.float64]:
    return numpy.array(oklab)


def _to_rgb_array(arr: npt.NDArray[ColourFloat]) -> npt.NDArray[numpy.intp]:
    return numpy.clip(arr * 255, 0, 255).astype(int)


_oklab_to_rgb: Callable[[Oklab], npt.NDArray[numpy.intp]] = compose_left_to_right(
    _to_ndarray, Oklab_to_XYZ, XYZ_to_sRGB, _to_rgb_array
)

_oklab_to_oklch: Callable[[Oklab], npt.NDArray[ColourFloat]] = compose_left_to_right(
    _to_ndarray, Oklab_to_Oklch
)


def oklch_string_to_oklab(
    oklch_str: str,
) -> Oklab:
    m: re.Match[str] | None = OKLCH_PATTERN.match(oklch_str)
    if m is None:
        raise ValueError(f"Invalid oklch string: {oklch_str}")
    oklch: npt.NDArray[numpy.float64] = numpy.array(
        [float(m.group(1)) / 100, float(m.group(2)), float(m.group(3))]
    )
    return compose_left_to_right(Oklch_to_Oklab, _to_oklab)(oklch)


def oklab_colored_text(oklab: Oklab, text: str) -> str:
    rgb: npt.NDArray[numpy.intp] = _oklab_to_rgb(oklab)
    return rgb_colored_text(rgb[0], rgb[1], rgb[2], text)


def color_to_oklab(color: str) -> Oklab:
    if OKLCH_PATTERN.match(color):
        return oklch_string_to_oklab(color)
    return hex_to_oklab(color)


def make_bluelight_hex_to_oklab(
    illuminant_xy: npt.NDArray[ColourFloat],
) -> Callable[[str], Oklab]:
    illuminant_XYZ: npt.NDArray[ColourFloat] = xy_to_XYZ(numpy.array(illuminant_xy))
    D65_XYZ: npt.NDArray[ColourFloat] = xy_to_XYZ(numpy.array([0.3127, 0.3290]))
    return compose_left_to_right(
        StandardRGB.create,
        partial(sRGB_to_XYZ, illuminant=illuminant_xy),
        lambda xyz: chromatic_adaptation(xyz, illuminant_XYZ, D65_XYZ),
        XYZ_to_Oklab,
        _to_oklab,
    )


def compute_xterm_color_difference(
    color: dict[str, str],
    hex_difference: Callable[[str], floating],
) -> XtermColorDifference:
    return XtermColorDifference(
        int(color["colorId"]),
        color["name"],
        color["hexString"],
        hex_difference(color["hexString"]),
    )


def _closest_xterm(
    hex_difference: Callable[[str], floating],
) -> list[XtermColorDifference]:
    with open("256-colors.json", "r") as f:
        iterator = json.loads(f.read()) | select(
            partial(compute_xterm_color_difference, hex_difference=hex_difference)
        )

        return heapq.nsmallest(
            5,
            iterator,
            key=lambda x: (x.delta_e, x.color_id),
        )


def print_xterm(hex_difference: Callable[[str], floating]) -> None:
    print("Closest xterm-256 colors:")
    for result in _closest_xterm(hex_difference):
        colorify = partial(xterm_colored_text, result.color_id)

        for k, v in result._asdict().items():
            print(f"  {k}: {v}")
        print(f"  {colorify(dummy_text)}")
        print()


def _closest_tailwind(target_oklab: Oklab) -> list[TailwindColorDifference]:
    skip_keys: set[str] = {"inherit", "current", "transparent"}

    tailwind_colors: dict[str, str | dict[str, str]]
    with open("tailwind-colors.json", "r") as f:
        tailwind_colors = json.load(f)

    tailwind_diffs: list[TailwindColorDifference] = []
    for name, shades in tailwind_colors.items():
        if name in skip_keys:
            continue
        if isinstance(shades, str):
            oklab = color_to_oklab(shades)
            d: floating = delta_E(numpy.array(target_oklab), numpy.array(oklab)).item()
            tailwind_diffs.append(TailwindColorDifference(name, "", shades, d))
        else:
            for shade, value in shades.items():
                oklab = color_to_oklab(value)
                d = delta_E(numpy.array(target_oklab), numpy.array(oklab)).item()
                tailwind_diffs.append(TailwindColorDifference(name, shade, value, d))

    return heapq.nsmallest(
        5,
        tailwind_diffs,
        key=lambda x: x.delta_e,
    )


def _tailwind_label(result: TailwindColorDifference) -> str:
    return f"{result.name}-{result.shade}" if result.shade else result.name


def print_tailwind(target_oklab: Oklab) -> None:
    print("Closest Tailwind colors:")
    for result in _closest_tailwind(target_oklab):
        oklab: Oklab = color_to_oklab(result.value)
        print(f"  name: {_tailwind_label(result)}")
        print(f"  value: {result.value}")
        print(f"  delta_e: {result.delta_e}")
        print(f"  {oklab_colored_text(oklab, dummy_text)}")
        print()


def make_hex_difference(
    target_oklab: Oklab,
    to_oklab: Callable[[str], Oklab],
) -> Callable[[str], floating]:
    def hex_difference(hex_color: str) -> floating:
        return delta_E(
            numpy.array(target_oklab),
            numpy.array(to_oklab(hex_color)),
        ).item()

    return hex_difference


def format_oklch(oklab: Oklab) -> str:
    oklch: npt.NDArray[ColourFloat] = _oklab_to_oklch(oklab)
    return f"oklch({oklch[0] * 100:.1f}% {oklch[1]:.4f} {oklch[2]:.3f})"


def format_hex(oklab: Oklab) -> str:
    rgb: npt.NDArray[numpy.intp] = _oklab_to_rgb(oklab)
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


@app.command()
def main(
    target_color: str = typer.Argument(..., help="Target color (hex or oklch)"),
    xterm_first: bool = typer.Option(
        False, help="Show xterm-256 results before Tailwind"
    ),
    bluelight: int | None = typer.Option(
        None, help="Color temperature in Kelvin for bluelight filter comparison"
    ),
) -> None:
    target_oklab: Oklab = color_to_oklab(target_color)
    target_hex: str = format_hex(target_oklab)
    target_oklch: str = format_oklch(target_oklab)

    if bluelight is not None:
        bluelight_filter_illuminant: npt.NDArray[ColourFloat] = CCT_to_xy_CIE_D(
            bluelight
        )
        bluelight_hex_to_oklab: Callable[[str], Oklab] = make_bluelight_hex_to_oklab(
            bluelight_filter_illuminant
        )
        hex_difference: Callable[[str], floating] = make_hex_difference(
            bluelight_hex_to_oklab(target_color), bluelight_hex_to_oklab
        )
    else:
        hex_difference = make_hex_difference(target_oklab, hex_to_oklab)

    print(f"Target: {target_hex} — {target_oklch}")
    print(f"  {oklab_colored_text(target_oklab, dummy_text)}", end="\n\n")

    if xterm_first:
        print_xterm(hex_difference)
        print_tailwind(target_oklab)
    else:
        print_tailwind(target_oklab)
        print_xterm(hex_difference)


if __name__ == "__main__":
    app()
