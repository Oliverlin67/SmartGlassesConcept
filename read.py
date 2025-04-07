import cv2
import mediapipe as mp
import numpy as np
import asyncio
import aiohttp
from datetime import datetime
from math import sqrt
from dataclasses import dataclass
#from filterpy.kalman import KalmanFilter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)

@dataclass
class Weather:
    name: str
    description: str

    def __repr__(self):
        return f"<Weather {self.name}({self.description})>"

@dataclass
class Report:
    dt: datetime
    temp: float
    feels_like: float
    temp_min: float
    temp_max: float
    humidity: int
    weather: Weather

    def __repr__(self):
        return f"<Report ({self.dt}) {self.temp}C (fl:{self.feels_like}C, [{self.temp_min}, {self.temp_max}]) {self.humidity}% with {self.weather}>"

class WeatherFetcher:
    def __init__(self, api_key: str, lat: str, lon: str):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon

    async def get_forecasts(self):
        result = []
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.openweathermap.org/data/2.5/forecast?lat={self.lat}&lon={self.lon}&units=metric&lang=zh_tw&cnt=5&appid={self.api_key}") as req:
                    forecasts = await req.json()
                    
                    for forecast in forecasts["list"]:
                        result.append(
                            Report(
                                dt=datetime.fromtimestamp(forecast["dt"]),
                                temp=forecast["main"]["temp"],
                                feels_like=forecast["main"]["feels_like"],
                                temp_min=forecast["main"]["temp_min"],
                                temp_max=forecast["main"]["temp_max"],
                                humidity=forecast["main"]["humidity"],
                                weather=Weather(
                                    name=forecast["weather"][0]["main"],
                                    description=forecast["weather"][0]["description"]
                                )
                            )
                        )
        except Exception as e:
            print(f"Error fetching forecasts: {e}")
        return result

    async def get_current(self):
        result = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://api.openweathermap.org/data/2.5/weather?lat={self.lat}&lon={self.lon}&units=metric&lang=zh_tw&appid={self.api_key}") as req:
                    report = await req.json()

                    result = Report(
                        dt=datetime.fromtimestamp(report["dt"]),
                        temp=report["main"]["temp"],
                        feels_like=report["main"]["feels_like"],
                        temp_min=report["main"]["temp_min"],
                        temp_max=report["main"]["temp_max"],
                        humidity=report["main"]["humidity"],
                        weather=Weather(
                            name=report["weather"][0]["main"],
                            description=report["weather"][0]["description"]
                        )
                    )
        except Exception as e:
            print(f"Error fetching current weather: {e}")
        return result

async def main():
    weatherFetcher = WeatherFetcher(
        api_key="c33f38c9ee6f1393f1b6479eb8deecb0",
        lat="25.0375198",
        lon="121.5636796"
    )

    cap = cv2.VideoCapture(1)
    forecasts = await weatherFetcher.get_forecasts()
    currentInfo = 'TIME'

    with mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Open failed")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                handsType = []
                for hand in results.multi_handedness:
                    handType = hand.classification[0].label
                    handsType.append(handType)

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    wrist = hand_landmarks.landmark[0]
                    index_base = hand_landmarks.landmark[5]
                    middle_base = hand_landmarks.landmark[9]
                    pinky_base = hand_landmarks.landmark[17]
                    middle_top = hand_landmarks.landmark[12]

                    h, w, _ = frame.shape
                    wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                    index_base_x, index_base_y = int(index_base.x * w), int(index_base.y * h)
                    pinky_base_x, pinky_base_y = int(pinky_base.x * w), int(pinky_base.y * h)
                    middle_base_x, middle_base_y = int(middle_base.x * w), int(middle_base.y * h)

                    src_pts = np.array([[191, 95], [5, 148], [0, 98], [43, 16]], dtype=np.float32)
                    dst_pts = np.array([[wrist_x, wrist_y], [index_base_x, index_base_y], [middle_base_x, middle_base_y], [pinky_base_x, pinky_base_y]], dtype=np.float32)

                    H, _ = cv2.findHomography(src_pts, dst_pts)

                    overlay = np.ones((190, 200, 3), dtype=np.uint8)
                    text = "-"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1
                    font_thickness = 2

                    print("H", H)

                    if sqrt((middle_top.x - wrist.x) ** 2 + (middle_top.y - wrist.y) ** 2) / sqrt((middle_base.x - wrist.x) ** 2 + (middle_base.y - wrist.y) ** 2) > 1:
                        if forecasts is None:
                            text = f"Fetching Weather"
                        elif len(forecasts) > 0:
                            text = f"Weather[n]{round(forecasts[0].temp*10)/10}*C[n]{forecasts[0].weather.name}"
                    else:
                        text = "Time[n]" + datetime.now().strftime("%H:%M:%S")

                    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                    text_x = int((200 - text_w / len(text.split('[n]'))) // 2)
                    text_y = (190 + text_h) // 2

                    for i, line in enumerate(text.split('[n]')):
                        cv2.putText(overlay, line, (text_x, text_y + i * (text_h+3)), font, font_scale, (0, 255, 0), font_thickness)

                    if handsType[idx] == 'Right':
                        overlay = cv2.flip(overlay, flipCode=1)
                        
                    if(index_base.y - pinky_base.y < 0):
                        overlay = cv2.flip(overlay, flipCode=0)

                    warped_overlay = cv2.warpPerspective(overlay, H, (w, h))

                    frame = cv2.addWeighted(frame, 1, warped_overlay, 1.5, 0)

                    cv2.imshow('Overlay', warped_overlay)

            cv2.imshow('Simulate', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())