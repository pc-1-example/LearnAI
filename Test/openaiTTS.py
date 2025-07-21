import asyncio

from openai import AsyncOpenAI
from openai.helpers import LocalAudioPlayer

openai = AsyncOpenAI()

input = """Whoa, dude… sounds like a gnarly situation with your card. But hey, no worries, I got you.\n\nLemme just pull up your account real quick… alright, looks like the system flagged some charges—probably thought they were, like, suspicious or somethin'. Super lame, I know.\n\nBut good news, my friend! I can clear that up right now. Just gotta verify a couple things, and boom—you'll be back in business, ridin' that wave of sweet, sweet purchases.\n\nHang tight, take a deep breath… we'll have you sorted in no time. Sound good, dude?"""

instructions = """Voice: Laid-back, mellow, and effortlessly cool, like a surfer who's never in a rush.\n\nTone: Relaxed and reassuring, keeping things light even when the customer is frustrated.\n\nSpeech Mannerisms: Uses casual, friendly phrasing with surfer slang like dude, gnarly, and boom to keep the conversation chill.\n\nPronunciation: Soft and drawn-out, with slightly stretched vowels and a naturally wavy rhythm in speech.\n\nTempo: Slow and easygoing, with a natural flow that never feels rushed, creating a calming effect."""

async def main() -> None:

    async with openai.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="coral",
        input=input,
        instructions=instructions,
        response_format="pcm",
    ) as response:
        await LocalAudioPlayer().play(response)

if __name__ == "__main__":
    asyncio.run(main())