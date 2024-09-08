# Video Scripts

## Video 1: What is an algorithm and how can you measure it?

**Script:** Generally speaking, an algorithm is a list of instructions, organized to complete a specific task. For example, a cake recipe is an algorithm. So is a workout routine for building muscle. In the context of coding problems, an algorithm is a sequence of computer instructions. When you create a computer algorithm, it is important to consider its efficiency. You can measure computer algorithm efficiency through time complexity and space complexity. Time and space complexity are the amount of time required and extra memory consumed as the problem size grows.

Here is an example of a coding problem:

given an array an unsored of positive integers like this one: [1,2,3,4,5,1,5,11] find the largest value. You can solve this with a simple for loop

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

**Disclaimer** Yes, you can also solve it like this **max(nums)** and get the same answer

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
