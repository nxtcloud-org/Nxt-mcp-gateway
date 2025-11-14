from mcp.server.fastmcp import FastMCP
from datetime import datetime
import pytz
from typing import Optional

# FastMCP 서버 초기화
mcp = FastMCP("TimeService")


@mcp.tool()
def get_current_time(timezone: Optional[str] = "Asia/Seoul") -> str:
    """
    지정된 타임존의 현재 시간을 반환합니다.

    Args:
        timezone: 타임존 (기본값: Asia/Seoul)

    Returns:
        현재 시간 정보
    """
    try:
        tz = pytz.timezone(timezone)
        current_time = datetime.now(tz)
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        return f"Current time in {timezone} is: {formatted_time}"
    except pytz.exceptions.UnknownTimeZoneError:
        return f"Error: Unknown timezone '{timezone}'. Please provide a valid timezone."
    except Exception as e:
        return f"Error getting time: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
