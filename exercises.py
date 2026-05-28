#punto 4
numbers=list(range(1,101))
diz={}
for number in numbers:
    diz[number]=number*number
#print(diz)
for key,value in diz.items():
    if key%3==0:
        print(f"{key}:{value}")
    elif key%5==0:
        print(f"{key}:{value}")
#punto5 
numbers=list(range(1,101))
roots={}
for number in numbers:
    roots[number]=number**0.5
square_nd_roots={}
for number in numbers:
    square_nd_roots[number]={"square":diz[number],"root":roots[number]}
print("\n")
print(square_nd_roots[9])
print("\n")
#punto 6

for keys,values in square_nd_roots.items():
    print(f"{keys}:{values}")

