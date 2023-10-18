print("set a, followed by 'enter'")
a = int(input())

print("set b, followed by 'enter'")
b = int(input())

print("set c, followed by 'enter'")
c = int(input())

print("set d, followed by 'enter'")
d = int(input())

print("set r, followed by 'enter'")
r = int(input())

print("a = " , a)
print("b = " , b)
print("c = " , c)
print("d = " , d)
print("r = " , r)

r34  = r+34
denomleft = r34*3
leftside = 4 / denomleft

abc = (a + b * c)
middle = 9 * abc


abd = a+ b*d
aplus2 = a+2
numerright = d*aplus2
right = numerright/abd
#please don't put in zero I don't know how to error handle yet

final = leftside- middle + 3 + right
print("Final output with given values is: ", final)