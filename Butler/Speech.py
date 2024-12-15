import pyttsx3
from collections import deque
from datetime import datetime
import threading
# The speech module of the Butler. Allows it to say things outloud and greet people.

class Speaker:

    def __init__(self):
        self.engine = pyttsx3.init()
        self.thingsToSay = deque()
        # self.currentUser = "Test User"
    
    def sayEverythingInQueue(self):
        while self.thingsToSay:       
            self.say(self.thingsToSay.pop())


    def say(self, phrase):
        self.engine.say(phrase)
        threading.Thread(target=self.engine.runAndWait).start()

    def greet(self, userName):
        timeOfDay = self.checkTimeOfDay()
        self.engine.say("Good " + timeOfDay + " " + userName)
        self.engine.say("How is your day going?")
        self.engine.say("Hold that thought, I still can't answer.")
        threading.Thread(target=self.engine.runAndWait).start()

        # self.engine.runAndWait()

    def checkTimeOfDay(self):
        time = datetime.now()
        hour = int(time.strftime("%H"))
        stageOfDay ="day"
        if(hour > 6 and hour <=11):
            stageOfDay = "morning"
        elif (hour > 11 and hour <=18):
            stageOfDay = "afternoon"
        elif ((hour > 18 and hour <=24) or (hour > 00 and hour <=6) ):
            stageOfDay = "evening"

        return stageOfDay