# Video Scripts

## Video 1: What is an algorithm and how can you measure it?

**Script:** An algorithm is a list of instructions, organized to complete a specific task. For example, a cake recipe is an algorithm. So is a workout routine for building muscle. In the context of coding problems, an algorithm is a sequence of computer instructions. When you create a computer algorithm, it is important to consider its efficiency. 

You can measure computer algorithm efficiency through time complexity and space complexity. Time and space complexity are the amount of time required and extra memory consumed as the problem size grows.

Here is an example of a coding problem and an algorithm that you can use to solve it

given an array of unsorted positive integers like this one: [1,2,3,4,5,1,5,11] find the largest value. 

You can solve this problem with a simple for loop

```python
def findLargestValue(nums):
  maxValue = -1
  for num in nums:
    if num > maxValue:
      maxValue = num
  return maxValue
```
In this case, the time complexity is O(n)
And the space complexity is O(1)

**Disclaimer** Yes, I am aware that you can also solve it like this **max(nums)** and get the same answer

## Video 2: What is Big-O Notation?

**Script:** Big-O notation is a way to simplify the worst case performance of an algorithm. For example if the actual runtime of an algorithm is can be described with the following function:

f(n) = 2 * n ^ 2  + n + 10, the corresponding Big O notation would be O(n^2) 

Here are some simple algorithms and their corresponding space and time complexity represented using Big-O notation 

```python3

# Time: O(1)
# Space: O(1)
def increment(number):
  return number + 1

# Time: O(n)
# Space: O(n)
def hasDuplicates(nums):
  mySet = set()
  for num in nums:
    if num in mySet:
      return True
    else:
      mySet.add(num)
  return False

# Time: O(log(n))
# Space: O(1)
def binarySearch(nums, target):
  startIdx, endIdx = 0, len(nums) - 1
  while startIdx <= endIdx:
    midIdx = startIdx + (endIdx - startIdx) // 2
    mid = nums[midIdx]
    if mid == target:
      return midIdx
    elif mid > target:
      endIdx = midIdx - 1
    else:
      startIdx = midIdx + 1
    return -1
```

## Video 3: What is a data structure?

**Script:** Data structures organize computer data in various ways. They have varying strengths and weaknesses that can be leveraged to create efficient algorithms. Here are a few useful data structures and how to create them in Python3:

1. String # store a sequence of characters
2. List # O(1) access
3. Map
4. Set
5. Deque # O(1) insert/delete
6. Graph # Represent nodes and vertices
7. Binary Tree # 
8. Linked list
9. Heap
10. Trie

In the next 10 videos, I'll cover the basics API of each of these 

## Video 4: How do Strings work in Python?

**Script:** Strings are data structures that store a sequence characters. Here is how to manipulate strings in Python:

```python3
1. Create a strings
2. Concatenate strings
3. Iterate over string
4. Convert string to ascii
5. Convert string into an array
6. Access element
7. Copy element
8. isalph()
9. isdigit
10. Starts with
11. Search list "i" in "string"
12. Get length
```

## Video 5: How do Lists work in Python?

**Script:** Lists/Arrays are extremely important data structures. Like strings they store sequences of data, but they are not limited to characters. 

```python3
1. create a list
2. append element
3. insert element
4. remove element
5. find min/max
6. sort
7. slice
8. reverse
9. Iterate over
10. copy
11. Iterate over list
12. Get length
```
## Video 6: How do Deques work in Python?

**Script:** A deque or double ended queue is another linear data structures. Because it allows you to insert and remove from either side of the in constant time, they are useful for situations where a stack or traditional queue are needed

1. Create a stack
2. Create a queue
3. Add values
4. pop
5. popleft
6. appendleft
7. Convert to list
8. Iterate over 

## Video 7: How do Sets work in Python?

**Script:** A set is another linear data structure that, like arrays, stores sequential data. The main difference is, a set can only store unique values. Because the values in a set are unique, they can be searched for in constant time. 

1. Create a set
2. Add values
3. Add duplicate values
4. Search for value
5. Remove value
6. Convert list to set
7. Convert set to list
8. Iterate over

## Video 8: How do Maps work in Python?
## Video 9: How can you make a Graph (Adj List) in Python?
## Video 10: How do you make a Binary Tree in Python?
## Video 11: How do you make a Linked List in Python?
## Video 12: How do you make a Heap in Python?
## Video 13: How do you make a Trie in Python?
## Video 14: How do you create a custom sort in Python?

## Video 15: How do you MergeSort/Quicksort in Python?
## Video 16: How do you reverse a Linked List?
## Video 17: How do you perform level order traversal on a tree / graph?
## Video 18: How do you determine the height of a binary tree?
## Video 19: How do you convert a tree to a graph?
## Video 20: How do you binary search a sorted list? How do modify to find the min/max?
## Video 21: How do you bfs (sshortest path) and dfs on a graph?
## Video 22: How do you convert edges to adjList (undirected, directed)?
## Video 23: How do you topo sort a graph?
## Video 24: How do you check for cycles in a graph?
## Video 25: How do you find the shortest path in a weighted graph?
## Video 26: How do you create a trie for scratch?
## Video 27: What are some useful bit manipulation patterns for coding problems?
## Video 28: How do you calculate Permutations, Combinations and Subsets programatically?
## Video 29: What are some examples of sliding window variations?
## Video 30: What are the variations of intervals in coding problems?

