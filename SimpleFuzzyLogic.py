#FUZZY LOGIC

def union(A, B):
    return {x: max(A.get(x, 0), B.get(x, 0)) for x in set(A) | set(B)}

def intersection(A, B):
    return {x: min(A.get(x, 0), B.get(x, 0)) for x in set(A) & set(B)}

def complement(A, precision=10):
    return {x: round(1 - A[x], precision) for x in A}

def difference(A, B):
    return {x: A.get(x, 0) * (1 - B.get(x, 0)) for x in set(A)}

# Membership values for sets A and B
A = {1: 0.1, 2: 0.3, 3: 0.5, 4: 0.7, 5: 0.9}
B = {3: 0.6, 4: 0.7}
print("First Fuzzy Set is : ",A)
print("Second Fuzzy Set is : ",B)
# Perform operations
union_result = union(A, B)
intersection_result = intersection(A, B)
complement_result = complement(A)
complement_result1 = complement(B)
difference_result = difference(A, B)

print("\nUnion : ", union_result)
print("Intersection : ", intersection_result)
print("Complement of A : ", complement_result)
print("Complement of B : ",complement_result1)
print("Difference (A - B) : ", difference_result)
