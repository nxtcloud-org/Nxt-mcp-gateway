from mcp.server.fastmcp import FastMCP
import requests
import os

# FastMCP 서버 초기화
mcp = FastMCP("WeatherService")

# OpenWeatherMap API 설정
# 참고: API 키는 생성 후 활성화까지 10분~2시간 소요됨
API_KEY = os.getenv("WEATHER_API_KEY", "08b906c2d7a625498bfd4b48b91f1faf")
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


@mcp.tool()
def get_weather(city: str = "Seoul") -> str:
    """
    도시의 현재 날씨를 조회합니다.

    Args:
        city: 도시명 (영문, 예: Seoul, Busan, Tokyo)

    Returns:
        날씨 정보 문자열
    """
    try:
        # API 요청
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric",  # 섭씨 온도
            "lang": "kr",  # 한국어 설명
        }

        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 데이터 추출
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        weather_desc = data["weather"][0]["description"]
        wind_speed = data["wind"]["speed"]

        # 결과 포맷팅
        result = f"""🌤️ **{city} 날씨**

🌡️ **온도**: {temp}°C (체감 {feels_like}°C)
☁️ **날씨**: {weather_desc}
💧 **습도**: {humidity}%
🌬️ **풍속**: {wind_speed} m/s"""

        return result

    except requests.exceptions.RequestException as e:
        return f"❌ 날씨 조회 실패: {str(e)}"
    except KeyError as e:
        return f"❌ 응답 파싱 오류: {str(e)}"
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"


@mcp.tool()
def get_forecast(city: str = "Seoul") -> str:
    """
    도시의 5일 예보를 조회합니다.

    Args:
        city: 도시명 (영문)

    Returns:
        5일 예보 정보
    """
    try:
        # 5일 예보 API
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {"q": city, "appid": API_KEY, "units": "metric", "lang": "kr"}

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # 3시간 간격 데이터에서 하루 1개씩만 추출 (12시 기준)
        result = f"📅 **{city} 5일 예보**\n\n"

        forecasts = data["list"]
        seen_dates = set()

        for item in forecasts:
            date = item["dt_txt"].split()[0]  # 날짜만 추출

            # 하루에 하나만 (중복 방지)
            if date in seen_dates or len(seen_dates) >= 5:
                continue

            seen_dates.add(date)

            temp = item["main"]["temp"]
            weather_desc = item["weather"][0]["description"]
            humidity = item["main"]["humidity"]

            result += f"📆 {date}\n"
            result += f"  🌡️ {temp}°C | ☁️ {weather_desc} | 💧 {humidity}%\n\n"

        return result.strip()

    except Exception as e:
        return f"❌ 예보 조회 실패: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
