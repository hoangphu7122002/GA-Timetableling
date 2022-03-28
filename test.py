str = "abc"
test = ""
for ele in str:
    test += ele
print(id(test) == id(str))