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
from numpy import longdouble, double
from colour import sRGB_to_XYZ, XYZ_to_Lab, delta_E
import json
import heapq
from pipe import select  # type: ignore[import]


P = ParamSpec("P")
T = TypeVar("T")
floating: TypeAlias = double | longdouble

dummy_text: str = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis a convallis ante,
vel volutpat risus. In vitae sapien a sapien ...
""".replace(
    "\n", ""
)


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


class XtermColorDifference(NamedTuple):
    color_id: int
    name: str
    delta_e: floating


def rgb_colored_text(r: int, g: int, b: int, text: str, background=False):
    return f"\033[{48 if background else 38};2;{r};{g};{b}m{text}\033[0;0m"


def xterm_colored_text(color_id: int, text: str, background=False):
    return f"\033[{48 if background else 38};5;{color_id}m{text}\033[0;0m"


if __name__ == "__main__":
    target_hex_string: str = "#66d9ef"
    target_sRGB: StandardRGB = StandardRGB.create(target_hex_string)

    print(
        rgb_colored_text(
            int(target_sRGB.red * 255),
            int(target_sRGB.green * 255),
            int(target_sRGB.blue * 255),
            f"{target_hex_string} target example:\n{dummy_text}",
        ),
        end="\n\n",
    )
    del target_sRGB

    hex_string_to_lab: Callable[
        [str], tuple[floating, floating, floating]
    ] = compose_left_to_right(StandardRGB.create, sRGB_to_XYZ, XYZ_to_Lab)

    def target_difference(hex_color: str):
        return delta_E(
            hex_string_to_lab(target_hex_string),
            hex_string_to_lab(hex_color),
            method="CIE 2000",
        )

    def compute_color_difference(color: dict[str, str]):
        return XtermColorDifference(
            int(color["colorId"]), color["name"], target_difference(color["hexString"])
        )

    ans: list[XtermColorDifference]
    with open("256-colors.json", "r") as f:
        ans = heapq.nsmallest(
            5,
            json.loads(f.read()) | select(compute_color_difference),
            key=lambda x: (x.delta_e, x.color_id),
        )

    for result in ans:
        colorify = partial(xterm_colored_text, result.color_id)

        for k, v in result._asdict().items():
            print(colorify(f"{k}: {v}"))
        print(colorify(dummy_text))
        print()
