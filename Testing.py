my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# Step 1: Create a list of tuples (element, original_index)
indexed_list = list(enumerate(my_list))

# Step 2: Sort the list of tuples based on the elements
sorted_indexed_list = sorted(indexed_list, key=lambda x: x[1])

# Step 3: Extract the sorted elements and their original indices
sorted_list = [element for index, element in sorted_indexed_list]
original_indices = [index for index, element in sorted_indexed_list]

print("Sorted list:", sorted_list)
print("Original indices:", original_indices)