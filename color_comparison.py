#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
from typing import NamedTuple, ParamSpec, TypeVar, Any, Callable, Self, TypeAlias
from functools import reduce
from numpy import longdouble, double
from colour import sRGB_to_XYZ, XYZ_to_Lab, delta_E
import json
import heapq
from pipe import select, Pipe, t  # type: ignore[import]


P = ParamSpec("P")
T = TypeVar("T")
floating: TypeAlias = double | longdouble


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


if __name__ == "__main__":
    target_hex_string: str = "#66d9ef"

    hex_string_to_lab: Callable[
        [str], tuple[floating, floating, floating]
    ] = compose_left_to_right(StandardRGB.create, sRGB_to_XYZ, XYZ_to_Lab)

    target_lab_value = hex_string_to_lab(target_hex_string)

    target_difference = lambda hex_color: delta_E(
        target_lab_value, hex_string_to_lab(hex_color), method="CIE 2000"
    )

    compute_color_difference = lambda color: XtermColorDifference(
        int(color["colorId"]),
        color["name"],
        target_difference(color["hexString"]),  # type: ignore[arg-type]
    )

    keep_five_smallest = Pipe(
        lambda i: heapq.nsmallest(5, i, key=lambda x: (x.delta_e, x.color_id))
    )

    ans: list[XtermColorDifference]
    with open("256-colors.json", "r") as f:
        ans = list(
            json.loads(f.read())
            | select(compute_color_difference)
            | keep_five_smallest
            | select(lambda result: result._asdict())
        )
    print(json.dumps(list(ans), indent=2))
