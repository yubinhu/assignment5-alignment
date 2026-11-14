import datetime
from zoneinfo import ZoneInfo
print(f"test-{datetime.datetime.now(ZoneInfo("America/Los_Angeles")):%Y%m%d-%H%M}")