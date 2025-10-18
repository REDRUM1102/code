array =[1,2,3,4,5,6,7,8,9]
array[0] = 'dict'
print('length of array is ',len(array)) 
# print(array[0:1])

for  x in array:
    print(x)

for (i, x) in enumerate(array):
    print(i, x, end=' ')
