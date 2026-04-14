from __future__ import annotations

import datetime as dt
import re


_ISO_FRACTION_RE = re.compile(
    r"^(?P<base>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})(?P<fraction>\.\d+)?(?P<tz>Z|[+-]\d{2}:\d{2})?$"
)


def parse_iso_datetime(value: str | dt.datetime) -> dt.datetime:
    """Parse ISO-8601 datetimes, including timestamps with nanosecond precision."""
    if isinstance(value, dt.datetime):
        return value

    normalized = str(value).strip()
    match = _ISO_FRACTION_RE.match(normalized)
    if match is None:
        return dt.datetime.fromisoformat(normalized.replace("Z", "+00:00"))

    fraction = match.group("fraction") or ""
    if len(fraction) > 7:
        fraction = fraction[:7]

    tz = match.group("tz") or ""
    if tz == "Z":
        tz = "+00:00"

    return dt.datetime.fromisoformat(f"{match.group('base')}{fraction}{tz}")
