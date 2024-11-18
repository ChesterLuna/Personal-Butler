import pyttsx3
from collections import deque
from datetime import datetime

# The speech module of the Butler. Allows it to say things outloud and greet people.

engine = pyttsx3.init()

thingsToSay = deque()
currentUser = "Test User"

def say(phrase):
    engine.say(phrase)
    engine.runAndWait()

def greet(userName):
    timeOfDay = checkTimeOfDay()
    engine.say("Good " + timeOfDay + " " + userName)
    engine.say("How is your day going?")
    engine.say("Hold that thought, I still can't answer.")
    engine.runAndWait()

def checkTimeOfDay():
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

def main():
    while True:        
        if (len(thingsToSay)):
            say(thingsToSay.pop())
        

if __name__ == "__main__":
    main()