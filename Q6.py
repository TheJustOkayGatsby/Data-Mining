
# take time in seconds and display it in minutes, seconds
print("Please enter time in seconds")
secs = int(input())
mins = int(secs/60)
remainingsecs = secs%60
print(secs , "seconds is equal to " , mins , "minutes and " , remainingsecs , "seconds.")