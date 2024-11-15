import pyttsx3
from collections import deque

# The speech module of the Butler. Allows it to say things outloud and greet people.

engine = pyttsx3.init()

thingsToSay = deque()
currentUser = "Test User"

def say(phrase):
    engine.say(thingsToSay.pop())
    engine.runAndWait()

def greet(userName):
    timeOfDay = checkTimeOfDay()
    engine.say("Good " + timeOfDay + " " + userName)
    engine.say("How is your day going?")
    engine.say("Hold that thought, I still can't answer.")
    engine.runAndWait()

def checkTimeOfDay():
    # TODO Should be either: Morning, Day, Afternoon, Evening.
    return "Day"

def main():
    while True:        
        if (len(thingsToSay)):
            say(thingsToSay.pop())
        

if __name__ == "__main__":
    main()