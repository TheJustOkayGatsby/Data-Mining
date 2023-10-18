#Declare two variables and initialise them both with integer values.
#Write a Python expression for ab² and display the result.

print("set value 1, followed by 'enter'")
first = int(input())

print("set value 2, followed by 'enter'")
second = int(input())

ab = first * second
#could this be done in one line?
squared = ab**2
print("if a = ", first , " and b = " , second , ",then ab =" , ab)
print("finally, ab**2 =" , squared)