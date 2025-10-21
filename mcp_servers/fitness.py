from mcp.server.fastmcp import FastMCP
from typing import Optional
import random

# Initialize FastMCP server with configuration
mcp = FastMCP(
    "FitnessCalculator",
    instructions="You are a fitness calculator that helps gym enthusiasts calculate weights with proper barbell considerations and motivational gym language.",
    host="0.0.0.0",
    port=8006,
)

# 헬창 명언 모음
GYM_QUOTES = [
    "봉은 조상님이 들어주냐?",
    "오늘도 철판과의 만남이군!",
    "무게는 거짓말하지 않는다!",
    "헬창의 길은 험난하다...",
    "철판 소리가 들려온다!",
    "오늘 안 하면 내일도 안 한다!",
    "근육은 배신하지 않는다!",
    "철판이 널 부르고 있다!",
    "헬창은 쉬는 날도 운동 생각한다!",
    "봉만 20kg인데 뭘 더 바라냐!",
]

MOTIVATION_QUOTES = [
    "💪 오늘도 화이팅!",
    "🔥 불타는 헬창 정신!",
    "⚡ 한계를 뛰어넘어라!",
    "🏋️‍♂️ 철판과 하나가 되어라!",
    "💯 완벽한 폼으로!",
    "🎯 목표를 향해!",
    "🚀 더 높이, 더 강하게!",
    "⭐ 오늘의 나를 이겨라!",
]


@mcp.tool()
async def calculate_total_weight(
    plates_weight: float, barbell_weight: Optional[float] = 20.0
) -> str:
    """
    헬창식 총 중량 계산기 - 봉 무게를 포함한 총 중량을 계산합니다.

    Args:
        plates_weight (float): 원판 무게 (kg)
        barbell_weight (float, optional): 봉 무게 (kg, 기본값: 20kg)

    Returns:
        str: 헬창식 계산 결과와 명언
    """
    try:
        total_weight = plates_weight + barbell_weight
        quote = random.choice(GYM_QUOTES)
        motivation = random.choice(MOTIVATION_QUOTES)

        result = f"""🏋️‍♂️ **헬창 계산기** 🏋️‍♂️

📊 **계산 결과:**
- 원판 무게: {plates_weight}kg
- 봉 무게: {barbell_weight}kg
- **총 중량: {total_weight}kg**

💬 **헬창 한마디:** "{quote}"

{motivation}

⚠️ **헬창 팁:** 봉 무게 {barbell_weight}kg는 기본이다! 원판만 {plates_weight}kg라고 하지 말고 총 {total_weight}kg라고 해야 진짜 헬창!"""

        return result.strip()

    except Exception as e:
        return f"❌ 계산 중 오류 발생: {str(e)}\n💪 다시 한번 도전해보자!"


@mcp.tool()
async def calculate_plates_needed(
    target_weight: float, barbell_weight: Optional[float] = 20.0
) -> str:
    """
    목표 중량에 필요한 원판 무게를 계산합니다.

    Args:
        target_weight (float): 목표 총 중량 (kg)
        barbell_weight (float, optional): 봉 무게 (kg, 기본값: 20kg)

    Returns:
        str: 필요한 원판 무게와 헬창 조언
    """
    try:
        if target_weight < barbell_weight:
            return f"❌ 목표 중량 {target_weight}kg가 봉 무게 {barbell_weight}kg보다 작습니다!\n💪 봉만으로도 {barbell_weight}kg인데 뭘 더 바라냐!"

        plates_needed = target_weight - barbell_weight
        quote = random.choice(GYM_QUOTES)
        motivation = random.choice(MOTIVATION_QUOTES)

        # 일반적인 원판 조합 제안
        plate_combinations = []
        remaining = plates_needed

        # 20kg 원판
        if remaining >= 40:  # 양쪽에 20kg씩
            twenties = int(remaining // 40) * 2
            plate_combinations.append(f"20kg 원판 {twenties}개")
            remaining = remaining % 40

        # 10kg 원판
        if remaining >= 20:  # 양쪽에 10kg씩
            tens = int(remaining // 20) * 2
            plate_combinations.append(f"10kg 원판 {tens}개")
            remaining = remaining % 20

        # 5kg 원판
        if remaining >= 10:  # 양쪽에 5kg씩
            fives = int(remaining // 10) * 2
            plate_combinations.append(f"5kg 원판 {fives}개")
            remaining = remaining % 10

        # 2.5kg 원판
        if remaining >= 5:  # 양쪽에 2.5kg씩
            twos = int(remaining // 5) * 2
            plate_combinations.append(f"2.5kg 원판 {twos}개")
            remaining = remaining % 5

        # 1.25kg 원판
        if remaining > 0:
            ones = int(remaining / 1.25) * 2
            if ones > 0:
                plate_combinations.append(f"1.25kg 원판 {ones}개")

        result = f"""🎯 **목표 중량 달성 계산기** 🎯

📊 **계산 결과:**
- 목표 총 중량: {target_weight}kg
- 봉 무게: {barbell_weight}kg
- **필요한 원판 무게: {plates_needed}kg**

🏋️‍♂️ **추천 원판 조합:**
{chr(10).join(f"- {combo}" for combo in plate_combinations) if plate_combinations else "- 봉만 사용하면 됩니다!"}

💬 **헬창 한마디:** "{quote}"

{motivation}

⚠️ **헬창 조언:** 원판은 양쪽에 균등하게 달아야 한다! 안전이 최우선!"""

        return result.strip()

    except Exception as e:
        return f"❌ 계산 중 오류 발생: {str(e)}\n💪 포기하지 말고 다시 도전!"


@mcp.tool()
async def calculate_1rm(
    weight: float, reps: int, barbell_weight: Optional[float] = 20.0
) -> str:
    """
    1RM (1회 최대 중량)을 계산합니다. Epley 공식 사용.

    Args:
        weight (float): 들어올린 총 중량 (kg)
        reps (int): 반복 횟수
        barbell_weight (float, optional): 봉 무게 (kg, 기본값: 20kg)

    Returns:
        str: 1RM 계산 결과와 헬창 격려
    """
    try:
        if reps <= 0:
            return "❌ 반복 횟수는 1 이상이어야 합니다!\n💪 최소한 1번은 들어야지!"

        if reps == 1:
            one_rm = weight
        else:
            # Epley 공식: 1RM = weight × (1 + reps/30)
            one_rm = weight * (1 + reps / 30)

        plates_only = weight - barbell_weight
        one_rm_plates = one_rm - barbell_weight

        quote = random.choice(GYM_QUOTES)
        motivation = random.choice(MOTIVATION_QUOTES)

        # 1RM 등급 평가 (일반적인 기준)
        if one_rm >= 200:
            grade = "🏆 괴물급"
        elif one_rm >= 150:
            grade = "💪 고수급"
        elif one_rm >= 100:
            grade = "🔥 중급자"
        elif one_rm >= 60:
            grade = "⭐ 초급자"
        else:
            grade = "🌱 입문자"

        result = f"""🏆 **1RM 계산기** 🏆

📊 **입력 정보:**
- 들어올린 중량: {weight}kg (원판 {plates_only}kg + 봉 {barbell_weight}kg)
- 반복 횟수: {reps}회

📈 **계산 결과:**
- **예상 1RM: {one_rm:.1f}kg** (원판 {one_rm_plates:.1f}kg + 봉 {barbell_weight}kg)
- 등급: {grade}

💬 **헬창 한마디:** "{quote}"

{motivation}

⚠️ **헬창 주의사항:** 1RM 도전은 반드시 스포터와 함께! 안전이 최우선이다!"""

        return result.strip()

    except Exception as e:
        return f"❌ 계산 중 오류 발생: {str(e)}\n💪 실패는 성공의 어머니다!"


@mcp.tool()
async def get_gym_motivation() -> str:
    """
    헬창 명언과 동기부여 메시지를 제공합니다.

    Returns:
        str: 랜덤 헬창 명언과 동기부여
    """
    quote = random.choice(GYM_QUOTES)
    motivation = random.choice(MOTIVATION_QUOTES)

    tips = [
        "폼이 완벽해야 진짜 헬창이다!",
        "꾸준함이 최고의 보충제다!",
        "오늘의 고통이 내일의 근육이다!",
        "철판 소리가 최고의 BGM이다!",
        "휴식도 운동의 일부다!",
        "단백질 섭취를 잊지 마라!",
        "스쿼트, 데드리프트, 벤치프레스 - 빅3는 기본!",
        "진짜 헬창은 봉 무게도 계산한다!",
    ]

    tip = random.choice(tips)

    result = f"""💪 **오늘의 헬창 동기부여** 💪

� **명언:** "{quote}"

{motivation}

💡 **헬창 팁:** {tip}

🏋️‍♂️ **기억하자:** 봉은 조상님이 들어주지 않는다! 20kg는 네가 들어야 한다!"""

    return result.strip()


if __name__ == "__main__":
    # Start the MCP server with stdio transport
    mcp.run(transport="stdio")
