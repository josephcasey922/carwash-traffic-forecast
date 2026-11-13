"""One-line, idempotent fix for historical 6 PM local forecast issuance dates.

Usage: python apply_asof_dst_fix.py
Only changes the exact affected asof.issue_time return expression. If the
source has changed, fails instead of overwriting an unknown version.
"""
from pathlib import Path


def apply(path: Path) -> str:
    source = path.read_text(encoding='utf-8')
    old='return pd.Timestamp(local_date).tz_localize(ZoneInfo(tz)) + pd.Timedelta(hours=hour)'
    new='return (pd.Timestamp(local_date) + pd.Timedelta(hours=hour)).tz_localize(ZoneInfo(tz))'
    if old in source:
        if source.count(old)!=1:
            raise RuntimeError('Unexpected repeated issuance-time code; no files changed')
        path.write_text(source.replace(old,new),encoding='utf-8')
        return f'Patched {path}; 6 PM now means 6 PM local time on DST Sundays.'
    if new in source or 'wall_clock.tz_localize(ZoneInfo(tz))' in source:
        return f'{path} already uses a daylight-saving-safe wall-clock issuance.'
    raise RuntimeError(f'Cannot find expected issue_time function in {path}; no files changed. '
                       'Please inspect asof.py manually before proceeding.')


if __name__=='__main__':
    print(apply(Path('asof.py')))
