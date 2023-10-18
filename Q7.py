#prompt name, age, gross pay and tax rate, then output name, age, gross pay , tax rate AND NET PAY
print("please enter name, followed by 'enter'")
name = input()

print("please enter age, followed by 'enter'")
age = int(input())

print("please enter gross pay, followed by 'enter'")
gpay = float(input())

print("please enter taxation rate as a decimal(e.g 12% = .12), followed by 'enter'")
tax = float(input())
# would like to know how to take a percentage as input

taxamt = float(gpay*tax)
npay = float(gpay - taxamt)

print("name = " , name)
print("age = " , age)
print("Gross pay = " , gpay)
print("tax rate = " , tax)
print("tax deducted = " , taxamt)
print("Net take-home pay = " , npay)
