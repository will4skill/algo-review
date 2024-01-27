# algo-review
Popular Algorithm Problems

## Python3 Cheatsheet
```python3
# https://www.valentinog.com/blog/python-for-js/
# https://neetcode.io/courses/lessons/python-for-coding-interviews
# https://www.w3schools.com/python/python_lists_comprehension.asp
# https://www.geeksforgeeks.org/python-using-2d-arrays-lists-the-right-way/
# https://www.pythoncheatsheet.org/cheatsheet/string-formatting
# https://www.interviewbit.com/python-cheat-sheet/#string-manipulation-in-python
# https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Map
# https://ioflood.com/blog/python-min-function-guide-uses-and-examples/

# floor division: 
25 // 2 #=> 12 JS: Math.floor(25/2)
a = 2
a += 1 #=> does not return anything, increments a

# Comparison operators:
> # greater than  
< # less than  
>= # greater than or equal to 
<= # less than or equal to  
== # equal  
!= # not equal  

# Logical operators: 
and 
or 
not

# Primative (Value) Types (immutable, e.g., operations on string return new string)
float
int
string
boolean # True, False
None # JS: null

# Complex (Reference) Types (all mutable except tuples)
list
dictionary
set
tuple # NOT MUTABLE 
deque # from collections import deque
hashSet 
minHeap # import heapq

#################################################################################################
# If-statements
# If statements don't need parentheses 
# or curly braces.
n = 1
if n > 2:
    n -= 1
elif n == 2:
    n *= 2
else:
    n += 2

# Parentheses needed for multi-line conditions.
# and = &&
# or  = ||
n, m = 1, 2
if ((n > 2 and 
    n != m) or n == m):
    n += 1

#################################################################################################
# Loops
n = 5
while n < 5:
    print(n)
    n += 1

# Looping from i = 0 to i = 4
for i in range(5):
    print(i)

# Looping from i = 2 to i = 5
for i in range(2, 6):
    print(i)

# Looping from i = 5 to i = 2
for i in range(5, 1, -1):
    print(i)
#################################################################################################
# Math
# Division is decimal by default
print(5 / 2) #=> 2.5
# Double slash rounds down
print(5 // 2) #=> 2
# CAREFUL: most languages round towards 0 by default, so negative numbers will round down
print(-3 // 2) #=> -2
# A workaround for rounding towards zero, is to use decimal division and then convert to int.
print(int(-3 / 2)) #=> -1
# Modding is similar to most languages
print(10 % 3) #=> 1
# Except for negative values
print(-10 % 3) #=> 2
# To be consistent with other languages modulo
import math
from multiprocessing import heap
print(math.fmod(-10, 3)) #=> -1.0

# More math helpers
print(math.floor(3 / 2)) #=> 1
print(math.ceil(3 / 2)) #=> 2
print(math.sqrt(2)) #=> 1.4142135623730951
print(math.pow(2, 3)) #=> 8.0

# Max / Min Int
float("inf") # JS: Number.MAX_VALUE
float("-inf") # JS: Number.MIN_VALUE

# Python numbers are infinite so they never overflow
print(math.pow(2, 200))

# But still less than infinity
print(math.pow(2, 200) < float("inf"))    

min(1, 2, 3) #=> 1
min([2, 3, 4]) #=> 2
min(['apple', 'banana', 'cherry'], key=len) #=> 'apple'
min({'apple': 1, 'banana': 2, 'cherry': 3}) #=> Output: 'apple'
min([10, 20, 30], [5, 15, 25]) #=> [5, 15, 25]

abs(-5) #=> 5


#################################################################################################
# Lists
my_list = ["a", None, 44, True, "f"]
my_list.append(44)
my_list.remove('f')
print(my_list) #=> ["a", None, 44, True, 44]
len(my_list) #=> 5
first_list = ["a", "b", "c"]
second_list = [1, 2, 3]
first_list + second_list #=> ["a", "b", "c", 1, 2, 3]
my_list.pop() #=> popped item
my_list.insert(1, 99) #=> insert 99 at index 1, returns None
my_list[0] #=> first item
my_list[-1] #=> last item
my_list[-2] #=> second to last item
length = 3
nums = [0]*length #=> [0, 0, 0]
nums[0:2] #=> sublist including first idx but not last [0, 0]
a, b, c = [1, 2, 3] # unpacking JS: [a, b, c] = [1, 2, 3]
bList = [1,2,3]
aList = bList.copy() # Shallow Copy

# https://stackoverflow.com/questions/509211/how-slicing-in-python-works
a = [1,2,3]
a[start:stop]  # items start through stop-1
a[start:]      # items start through the rest of the array (Tim note: past end, => [])
a[:stop]       # items from the beginning through stop-1
a[:]           # a copy of the whole array

# Loop JS: for (const num of nums) console.log(num)  
# ** Note: string iteration works the same **
for num in nums:
    print(num)

# Loop with index
for idx in range(len(nums)):
    print(nums[idx])

# Loop with index and value
for idx, num in enumerate(nums):
    print(idx, num)

# Loop through multiple lists at the same time with unpacking
nums1 = [1, 3, 5]
nums2 = [2, 4, 6]
for n1, n2 in zip(nums1, nums2):
    print(n1, n2)

# Reverse
nums.reverse()

# Sort
nums.sort() # Smallest to largest or alphabetical
nums.sort(reverse=True)

# Sum
sum(nums)

# Custom Sort (sort by length of string, shortest to longest)
arr = ["bob", "alice", "jane", "doe"]
arr.sort(key=lambda x: len(x)) # JS: arr.sort((a, b) => a.length - b.length)
print(arr)

# List comprehension
fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = [x.upper() for x in fruits if "a" in x] 
# JS: newList = fruits.filter((x) => x.includes('a')).map((x) => x.toUpperCase());
print(newlist) #=> ['APPLE', 'BANANA', 'MANGO']

# 2-D list initialization with comprehension
rows, cols = (5, 5)
arr = [[0 for i in range(cols)] for j in range(rows)] #=> my2DArray = Array(rows).fill().map(() => Array(cols).fill(0));
print(arr) #=> [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

arr = [[0] * cols for i in range(rows)]
print(arr)

#################################################################################################
# Strings
# Strings are similar to arrays
s = "abc"
print(s[0:2]) #=> ab

# But they are immutable
# s[0] = "A"

# So this creates a new string
s += "def"
print(s) #=> abcdef

"o" in "Florence" #=> True Note: you must search for string or get an exception JS: 

# Valid numeric strings can be converted
print(int("123") + int("123")) #=> 246

# And numbers can be converted to strings
print(str(123) + str(123)) #=> 123123

# In rare cases you may need the ASCII value of a char
print(ord("a")) #=> 97 JS: "a".charCodeAt(0);
print(ord("b")) #=> 98

# Combine a list of strings (with an empty string delimitor)
strings = ["ab", "cd", "ef"]
print("".join(strings)) #=> abcdef JS: strings.join("")
"ab,cd,ef".split(',') #=> ["ab", "cd", "ef"] Note: " " is the default delim
list("abcdef") #=> ['a', 'b', 'c', 'd', 'e', 'f']

a = 5
b = 10
f'Five plus ten is {a + b} and not {2 * (a + b)}.' #=> 'Five plus ten is 15 and not 30.'
# JS: `Five plus ten is ${a + b} and not ${2 * (a + b)}.`

"".isspace() # Returns True if all characters in string are whitespaces
"".isalnum() # Returns True if given string is alphanumeric
"".isalpha() # Returns True if given character is alphabet

for c in "string":
    #do something with c

#################################################################################################
# Queues
from collections import deque

queue = deque()
queue.append(1) # JS: queue.push()
queue.append(2)
print(queue) #=> deque([1, 2])

queue.popleft() #=> 1 JS: queue.shift()
print(queue) #=> deque([2])

queue.appendleft(1) # JS: queue.unshift(1)
print(queue) #=> deque([1, 2])

queue.pop() #=> 2 JS: queue.pop()
print(queue) #=> deque([1])

#################################################################################################
# HashSets
mySet = set() # JS: mySet = new Set()

mySet.add(1) # JS: mySet.add(1)
mySet.add(2)
print(mySet) #=> {1, 2}
print(len(mySet)) #=> 2 JS: mySet.size

print(1 in mySet) #=> True JS: mySet.has(1)
print(2 in mySet) #=> True
print(3 in mySet) #=> False

mySet.remove(2) #=> **THROWS ERROR** JS: mySet.delete(2) 
print(2 in mySet)

# list to set
print(set([1, 2, 3])) # JS: new Set([1, 2, 3])

# Set comprehension
mySet = { i for i in range(5) }
print(mySet) #=> {0, 1, 2, 3, 4}

#################################################################################################
# HashMaps
myMap = {
    "alice": 88,
    "bob": 77
}
print(myMap) #=> {'alice': 88, 'bob': 77}
print(len(myMap)) #=> 2

myMap["alice"] = 80
print(myMap["alice"]) #=> alice

print("alice" in myMap) #=> True
myMap.pop("alice") #=> remove value JS: delete myMap.alice;
print("alice" in myMap) #=> False

myMap["random"] #=> **Throws Exception**
myMap.get("random") #=> None
myMap.get("random", "not found") #=> not found Returns default message, does nothing else

# Dict comprehension
myMap = { i: 2*i for i in range(3) }
print(myMap) #=> {0: 0, 1: 2, 2: 4}

# Looping through maps
myMap = { "alice": 90, "bob": 70 }

for key in myMap: 
    print(key, myMap[key])

for val in myMap.values():
    print(val)

for key, val in myMap.items(): # JS: for (const key in myMap) console.log(key, myMap[key])
    print(key, val)

myMap.clear() # myMap is empty now

# Dict unpacking works like spread 
initial = {"dont_touch": "my breil"}
next = {**initial, "dont_touch": "just a copy"}
print(next) #=> {'dont_touch': 'just a copy'}

a = {"name": "Juliana", "age": 33}
b = {"surname": "Crain", "city": "San Francisco"}
all = {**a, **b}
print(all) #=> {'name': 'Juliana', 'age': 33, 'surname': 'Crain', 'city': 'San Francisco'}

#################################################################################################
# Heaps
import heapq
# under the hood are arrays
minHeap = []
heapq.heappush(minHeap, 3)
heapq.heappush(minHeap, 2)
heapq.heappush(minHeap, 4)

# Min is always at index 0
print(minHeap[0]) #=> 2

while len(minHeap):
    print(heapq.heappop(minHeap)) #=> 2 3 4

# No max heaps by default, work around is
# to use min heap and multiply by -1 when push & pop.
maxHeap = []
heapq.heappush(maxHeap, -3)
heapq.heappush(maxHeap, -2)
heapq.heappush(maxHeap, -4)

# Max is always at index 0
print(-1 * maxHeap[0]) #=> 4

while len(maxHeap):
    print(-1 * heapq.heappop(maxHeap)) #=> 4 3 2

# Build heap from initial values
arr = [2, 1, 8, 4, 5]
heapq.heapify(arr)
while arr:
    print(heapq.heappop(arr)) #=> 1 2 4 5 8

#################################################################################################
# Tuples: are like arrays but immutable
my_tuple = ("vale", "Italy", 105)
print(my_tuple) #=> ('vale', 'Italy', 105)
my_tuple.count("Italy") #=> 1
my_tuple.index(105) #=> 2
print(my_tuple[0]) #=> vale
print(my_tuple[-1]) #=> 105
# Can't modify
# my_tuple[0] = 0 # *Throws Error*

# Can be used as key for hash map/set
myMap = { (1,2): 3 } # JS: object keys are strings, The keys in a JS Map() Any value (both objects and primitive values) may be used as either a key or a value (* Same with Set())
print(myMap[(1,2)]) #=> 3

mySet = set()
mySet.add((1, 2))
print((1, 2) in mySet) #=> True

# Lists can't be keys
# myMap[[3, 4]] = 5 # * Throws Error * 

#################################################################################################
# Functions
def myFunc(n, m):
    return n * m

print(myFunc(3, 4))

# Nested functions have access to outer variables
def outer(a, b):
    c = "c"

    def inner():
        return a + b + c
    return inner()

print(outer("a", "b"))

# Can modify objects but not reassign, unless using **nonlocal** keyword
def double(arr, val):
    def helper():
        # Modifying array works
        for i, n in enumerate(arr):
            arr[i] *= 2
        # will only modify val in the helper scope
        # val *= 2

        # this will modify val outside helper scope
        nonlocal val
        val *= 2
    helper()
    print(arr, val)

nums = [1, 2]
val = 3
double(nums, val)

def mul(a, b):
   return a * b
print(mul(3, 5)) #=> 15

mul = lambda a, b: a * b
print(mul(3, 5)) #=> 15


#################################################################################################
# Classes
class Person:
    def __init__(self, name, age): # Constructor
        self.name = name
        self.age = age

    def print_details(self):
        details = f"Name: {self.name} - Age: {self.age}"
        print(details)

tom = Person("Tom", 89)
tom.print_details() #=> Name: Tom - Age: 89
isinstance(tom, Person) #=> True

isinstance(9, int) #=> True
isinstance(tom, Person) #=> True
isinstance("caty", str) #=> True
type(9) is int #=> True
type(tom) is Person #=> True
type("caty") is str #=> True

# class Person {
#     constructor(name, age) {
#         this.name = name
#         this.age = age
#     }
#     printDetails() {
#         const details = `Name: ${this.name} - Age: ${this.age}`
#         console.log(details)
#     }
# }
# const tom = new Person("Tom", 44)
# console.log(tom instanceof Person) #=> true
```


## 1. Two Sum
**Reference:** https://leetcode.com/problems/two-sum/solutions/3619262/3-method-s-c-java-python-beginner-friendly/

**Description:** Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

**Constraints:** You may assume that each input would have exactly one solution, and you may not use the same element twice.

**Examples:** 
```python3
nums = [2,7,11,15], target = 9 #=> [0,1]
```

**Hint:** Load values into hash map, iterate over nums and check hash map for complement

```python3
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        numMap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in numMap:
                return [numMap[complement], i]
            numMap[nums[i]] = i
        return []  # No solution found
```
**Time:** O(n)
**Space:** O(n)

## 2. Best Time to Buy and Sell Stock
**Reference:** https://leetcode.com/problems/best-time-to-buy-and-sell-stock/

**Description:** You are given an array prices where prices[i] is the price of a given stock on the ith day. You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock. Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

**Constraints:** 1 <= prices.length <= 10^5, 0 <= prices[i] <= 10^4

**Examples:** 
```python3
prices = [7,1,5,3,6,4] #=> 5
prices = [7,6,4,3,1] #=> 0
```

**Hint:** Iterate from end, try each buy day

```python3
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        maxSeen = prices[-1]
        maxProfitValue = 0
        for i in range(len(prices) - 2, -1, -1):
            maxSeen = max(maxSeen, prices[i+1])
            maxProfitValue = max(maxProfitValue, maxSeen - prices[i])
        return maxProfitValue
```
**Time:** O(n)
**Space:** O(1)

## 3. Majority Element
**Reference:** https://leetcode.com/problems/majority-element/solutions/51712/python-different-solutions/

**Description:** Given an array nums of size n, return the majority element. The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.

**Constraints:** n == nums.length, 1 <= n <= 5 * 10^4, -10^9 <= nums[i] <= 10^9

**Examples:** 
```python3
nums = [3,2,3] #=> 3
nums = [2,2,1,1,1,2,2] #=> 2
```

**Hint:** Boyer Moore: Same element, inc count. Diff and count zero, pick new candidate. Diff and count > 0, decrement

```python3
class Solution:
    def majorityElement(self, nums):
        candidate, count = nums[0], 0
        for num in nums:
            if num == candidate: # Same number, increase count
                count += 1
            elif count == 0: # diff number and count zero, pick new candidate
                candidate, count = num, 1
            else: # diff number and count > 0 decrease candidate 
                count -= 1
        return candidate
```
**Time:** O(n)
**Space:** O(1)

## 4. Contains Duplicate
**Reference:** https://leetcode.com/problems/majority-element/solutions/51712/python-different-solutions/

**Description:** Given an integer array nums, return true if any value appears at least twice in the array, and return false if every element is distinct.

**Constraints:** 1 <= nums.length <= 10^5, -10^9 <= nums[i] <= 10^9

**Examples:** 
```python3
nums = [1,2,3,1] #=> true
nums = [1,2,3,4] #=> false
```

**Hint:** Use set or HashMap

```python3
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        mySet = set()
        for num in nums:
            if num in mySet:
                return True
            else:
                mySet.add(num)
        return False
```
**Time:** O(n)
**Space:** O(n)

## 5. Meeting Rooms
**Reference:** 
https://aaronice.gitbook.io/lintcode/sweep-line/meeting-rooms
https://github.com/neetcode-gh/leetcode/blob/main/python/0252-meeting-rooms.py

***Description:** Given an array of meeting time intervals consisting of start and end times[[s1,e1],[s2,e2],...](si< ei), determine if a person could attend all meetings.

**Constraints:** ??

**Examples:** 
```python3
[[0,30],[5,10],[15,20]] #=> false
[[7,10],[2,4]] #=> true
```

**Hint:** Sort and then compare each element

```python3
class Solution:
    def canAttendMeetings(self, intervals):
        intervals.sort(key=lambda i: i[0])
        for i in range(1, len(intervals)):
            i1 = intervals[i - 1]
            i2 = intervals[i]
            if i1[1] > i2[0]:
                return False
        return True
```
**Time:** O(nlog(n))
**Space:** O(1)

## 6. Move Zeroes
**Reference:** https://leetcode.com/problems/move-zeroes/solutions/562911/two-pointers-technique-python-o-n-time-o-1-space/

**Description:** Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements. Note that you must do this in-place without making a copy of the array. Follow up: Could you minimize the total number of operations done?

**Constraints:** 1 <= nums.length <= 10^4, -2^31 <= nums[i] <= 2^31 - 1

**Examples:** 
```python3
nums = [0,1,0,3,12] #=> [1,3,12,0,0]
nums = [0] #=> [0]
```

**Hint:** Fast and slow pointer, slow searches for zeros, swap fast and slow when zero found

```python3
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != 0 and nums[slow] == 0:
                nums[slow], nums[fast] = nums[fast], nums[slow]

            # wait while we find a non-zero element to
            # swap with you
            if nums[slow] != 0:
                slow += 1
```
**Time:** O(n)
**Space:** O(1)

## 7. Squares of a Sorted Array
**Reference:** https://leetcode.com/problems/move-zeroes/solutions/562911/two-pointers-technique-python-o-n-time-o-1-space/

**Description:** Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order. Follow up: Squaring each element and sorting the new array is very trivial, could you find an O(n) solution using a different approach?

**Constraints:** 1 <= nums.length <= 10^4, -10^4 <= nums[i] <= 10^4, nums is sorted in non-decreasing order.

**Examples:** 
```python3
nums = [-4,-1,0,3,10] #=> [0,1,9,16,100]
nums = [-7,-3,2,3,11] #=> [4,9,9,49,121]
```

**Hint:** Square everything, reverse the negative numbers (or just use two pointers), then use merge from merge sort

```python3
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        arr1 = []
        arr2 = []
        output = []

        for num in nums:
            if num < 0:
                arr1.insert(0,num**2)
            else:
                arr2.append(num**2)
        
        idx1 = 0
        idx2 = 0

        while idx1 < len(arr1) and idx2 < len(arr2):
            num1 = arr1[idx1]
            num2 = arr2[idx2]
            if num1 < num2:
                output.append(num1)
                idx1 += 1
            else:
                output.append(num2)
                idx2 += 1

        while idx1 < len(arr1):
            output.append(arr1[idx1])
            idx1 += 1

        while idx2 < len(arr2):
            output.append(arr2[idx2])
            idx2 += 1

        return output
```
**Time:** O(n)
**Space:** O(n)

## 8. Insert Interval
**Reference:** https://leetcode.com/problems/insert-interval/solutions/844549/python-super-short-simple-clean-solution-99-faster/

**Description:** You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval. Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary). Return intervals after the insertion.

**Constraints:** 0 <= intervals.length <= 10^4, intervals[i].length == 2, 0 <= starti <= endi <= 10^5, intervals is sorted by starti in ascending order, newInterval.length == 2, 0 <= start <= end <= 105

**Examples:** 
```python3
intervals = [[1,3],[6,9]], newInterval = [2,5] #=> [[1,5],[6,9]]
intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8] #=> [[1,2],[3,10],[12,16]]
```

**Hint:** 
Scenarios: 
1. The new interval is in the range of the prev interval
2. The new interval's range is before the prev interval
3. The new interval is after the range of the prev interval

```python3
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        result = []
        for interval in intervals:
			# the new interval is after the range of other interval, so we can leave the current interval baecause the new one does not overlap with it
            if interval[1] < newInterval[0]:
                result.append(interval)
            # the new interval's range is before the other, so we can add the new interval and update it to the current one
            elif interval[0] > newInterval[1]:
                result.append(newInterval)
                newInterval = interval
            # the new interval is in the range of the other interval, we have an overlap, so we must choose the min for start and max for end of interval 
            elif interval[1] >= newInterval[0] or interval[0] <= newInterval[1]:
                newInterval[0] = min(interval[0], newInterval[0])
                newInterval[1] = max(newInterval[1], interval[1])
        result.append(newInterval); 
        return result
```
**Time:** O(n)
**Space:** O(n)

## 9. 3Sum
**Reference:** https://leetcode.com/problems/3sum/solutions/593246/3sum/

**Description:** Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0. Notice that the solution set must not contain duplicate triplets.

**Constraints:** 3 <= nums.length <= 3000, -10^5 <= nums[i] <= 10^5

**Examples:** 
```python3
nums = [-1,0,1,2,-1,-4] #=> [[-1,-1,2],[-1,0,1]]
nums = [0,1,1] #=> []
nums = [0,0,0] #=> [[0,0,0]]
```

**Hint:** Sort arr, then for each entry, compare the left and right elements after the current one

```javascript
const threeSum = (nums) => {
	nums.sort((a,b) => a-b)
	const result=[]
	if(nums.length < 3) return result

  for(let i=0; i< nums.length; i++){
      let left = i+1
      let right = nums.length-1
      if(nums[i] === nums[i-1]) continue
      // Two Sum II
      while(left < right){
          const sum = nums[i] + nums[left] + nums[right]
          if(sum === 0){
              const arr=[nums[i], nums[left], nums[right]]
              result.push(arr)
              while(left < right && nums[left] === nums[left+1]){
                  left++
              }
              while(left < right && nums[right] === nums[right-1]){
                  right--
              }
              left++
              right--
          }else if(sum > 0){
              right--
          }else{
              left++
          }
      }
  }
  return result
};
```
**Time:** O(n^2)
**Space:** O(log n) to n, O(n) depending on the sorting algorithm


## 10. Product of Array Except Self
**Reference:** https://leetcode.com/problems/product-of-array-except-self/solutions/65622/simple-java-solution-in-o-n-without-extra-space/

**Description:** Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer. You must write an algorithm that runs in O(n) time and without using the division operation. Follow up: Can you solve the problem in O(1) extra space complexity? (The output array does not count as extra space for space complexity analysis.)

**Constraints:** 2 <= nums.length <= 10^5, -30 <= nums[i] <= 30, The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

**Examples:** 
```python3
nums = [1,2,3,4] #=> [24,12,8,6]
nums = [-1,1,0,-3,3] #=> [0,0,9,0,0]
```

**Hint:** Get the values to the right of each number, get the values to the left of each number, combine multiply the lefts and rights to create the final solution

```python3
def productExceptSelf(nums):
    n = len(nums)
    res = [0] * n
    res[0] = 1
    for i in range(1, n):
        res[i] = res[i - 1] * nums[i - 1]
    right = 1
    for i in range(n - 1, -1, -1):
        res[i] *= right
        right *= nums[i]
    return res
```
**Time:** O(n)
**Space:** O(n)

## 11. Combination Sum
**Reference:** https://leetcode.com/problems/permutations/discuss/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partioning)

**Description:** Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

**Constraints:** 1 <= nums.length <= 6, -10 <= nums[i] <= 10, All the integers of nums are unique.

**Examples:** 
```python3
nums = [1,2,3] #=> [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
nums = [0,1] #=> [[0,1],[1,0]]
nums = [1] #=> [[1]]
```

**Hint:** Backtracking, include or don't include each number until target = 0

```python3
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        result = []
        self.dfs(result, [], candidates, target, 0)
        return result
    
    def dfs(self, result, tempList, nums, target, idx):
        if target < 0: return
        if target == 0: 
            result.append(tempList.copy())
            return
        for i in range(idx, len(nums)):
            tempList.append(nums[i])
            self.dfs(result, tempList, nums, target - nums[i], i) 
            tempList.pop()
```
**Time:** O(2^n)
**Space:** O(n)

## 12. Merge Intervals
**Reference:** https://leetcode.com/problems/merge-intervals/solutions/350272/python3-sort-o-nlog-n/

**Description:** Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

**Constraints:** 1 <= intervals.length <= 10^4, intervals[i].length == 2, 0 <= starti <= endi <= 10^4

**Examples:** 
```python3
intervals = [[1,3],[2,6],[8,10],[15,18]] #=> [[1,6],[8,10],[15,18]]
intervals = [[1,4],[4,5]] #=> [[1,5]]
```

**Hint:** Sort the intervals, then iterate over them comparing curr to prev and merging if overlap

```python3
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()
        merged = []
        for i in range(len(intervals)):
            if merged == []:
                merged.append(intervals[i])
            else:
                previous_end = merged[-1][1]
                current_start = intervals[i][0]
                current_end = intervals[i][1]
                if previous_end >= current_start: # overlap
                    merged[-1][1] = max(previous_end,current_end)
                else:
                    merged.append(intervals[i])
        return merged
```
**Time:** O(nlog(n))
**Space:** O(n)

## 13. Sort Colors
**Reference:** https://leetcode.com/problems/sort-colors/solutions/26549/java-solution-both-2-pass-and-1-pass/

**Description:** Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue. We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively. You must solve this problem without using the library's sort function. Follow up: Could you come up with a one-pass algorithm using only constant extra space?

**Constraints:** n == nums.length, 1 <= n <= 300, nums[i] is either 0, 1, or 2.

**Examples:** 
```python3
nums = [2,0,2,1,1,0] #=> [0,0,1,1,2,2]
nums = [2,0,1] #=> [0,1,2]
```

**Hint:** Count each color, then insert them in order (2 pass), or 2 pointer (below)

```python3
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        p1, p2, index = 0, len(nums) - 1, 0
        while index <= p2:
            if nums[index] == 0:
                nums[index], nums[p1] = nums[p1], 0
                p1 += 1
            if nums[index] == 2:
                nums[index], nums[p2] = nums[p2], 2
                p2 -= 1
                index -= 1
            index += 1
```
**Time:** O(n)
**Space:** O(1)

## 14. Container With Most Water
**Reference:** https://leetcode.com/problems/container-with-most-water/description/
https://leetcode.com/problems/container-with-most-water/ (official)

**Description:** You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]). Find two lines that together with the x-axis form a container, such that the container contains the most water. Return the maximum amount of water a container can store. Notice that you may not slant the container.

**Constraints:** n == height.length, 2 <= n <= 10^5, 0 <= height[i] <= 10^4

**Examples:** 
```python3
height = [1,8,6,2,5,4,8,3,7] #=> 49
height = [1,1] #=> 1
```

**Hint:** Use two pointers. The area between two lines is limited by the shorter line. Fatter lines have more area (if height is equal). Create start and end pointers. Maintain global max. At each step, find new area and update global max if it is greater. Move the shorter line's ptr forward one step (because shorter is the limiter). Stop when ptrs converge

```javascript
const maxArea = (height) => {
  let maxarea = 0;
  let left = 0, right = height.length - 1;

  while (left < right) {
      const width = right - left;
      maxarea = Math.max(maxarea, Math.min(height[left], height[right]) * width);

      if (height[left] > height[right]) right--;
      else left++;
  }
  return maxarea;
};
```
**Time:** O(n)
**Space:** O(1)

## 15. Gas Station
**Reference:** https://leetcode.com/problems/gas-station/solutions/1706142/java-c-python-an-explanation-that-ever-exists-till-now/

**Description:** There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i]. You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations. Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique

**Constraints:** n == gas.length == cost.length, 1 <= n <= 10^5, 0 <= gas[i], cost[i] <= 10^4

**Examples:** 
```python3
gas = [1,2,3,4,5], cost = [3,4,5,1,2] #=> 3
gas = [2,3,4], cost = [3,4,3] #=> -1
```

**Hint:** Sim to maximum sub array. "if we run out of fuel say at some ith gas station. All the gas station between ith and starting point are bad starting point as well.
So, this means we can start trying at next gas station on the i + 1 station." If it is possible to make a round trip, the sum of all gas - the cost of all trips must be >= 0. Try starting from i = 0, if you reach a negative tank, start again at the next index

```python3
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n, total_surplus, surplus, start = len(gas), 0, 0, 0
        for i in range(n):
            total_surplus += gas[i] - cost[i]
            surplus += gas[i] - cost[i]
            if surplus < 0:
                surplus = 0
                start = i + 1
        return -1 if (total_surplus < 0) else start
```
**Time:** O(n)
**Space:** O(1)

## 16. Longest Consecutive Sequence
**Reference:** https://leetcode.com/problems/longest-consecutive-sequence/solutions/41057/simple-o-n-with-explanation-just-walk-each-streak/

**Description:** Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence. You must write an algorithm that runs in O(n) time.

**Constraints:** 0 <= nums.length <= 10^5, -10^9 <= nums[i] <= 10^9

**Examples:** 
```python3
nums = [100,4,200,1,3,2] #=> 4
nums = [0,3,7,2,5,8,4,6,0,1] #=> 9
```

**Hint:** Maybe Recursive DP. Maintain global max. For each number use a set to traverse as for as possible, once you can't, compare that local max with the global max distance, only travere + direction otherwise you'll need some kind of visited set

```python3
def longestConsecutive(self, nums):
    nums = set(nums)
    best = 0
    for x in nums:
        if x - 1 not in nums:
            y = x + 1
            while y in nums:
                y += 1
            best = max(best, y - x)
    return best
};
```
**Time:** O(n)
**Space:** O(n)

## 17. Rotate Array
**Reference:** https://leetcode.com/problems/rotate-array/solutions/54250/easy-to-read-java-solution/

**Description:** Given an integer array nums, rotate the array to the right by k steps, where k is non-negative. Follow up: Try to come up with as many solutions as you can. There are at least three different ways to solve this problem. Could you do it in-place with O(1) extra space?

**Constraints:** 1 <= nums.length <= 10^5. -2^31 <= nums[i] <= 2^31 - 1. 0 <= k <= 10^5

**Examples:** 
```python3
nums = [1,2,3,4,5,6,7], k = 3 #=> [5,6,7,1,2,3,4]
nums = [-1,-100,3,99], k = 2 #=> [3,99,-1,-100]
```

**Hint:** 
Steps:
1. Reverse entire nums
2. Reverse nums before k
3. Reverse nums k to end

```python3
public void rotate(int[] nums, int k) {
    k %= nums.length;
    reverse(nums, 0, nums.length - 1);
    reverse(nums, 0, k - 1);
    reverse(nums, k, nums.length - 1);
}

public void reverse(int[] nums, int start, int end) {
    while (start < end) {
        int temp = nums[start];
        nums[start] = nums[end];
        nums[end] = temp;
        start++;
        end--;
    }
}
```
**Time:** O(n)
**Space:** O(1)

## 18. Contiguous Array
**Reference:** https://leetcode.com/problems/rotate-array/solutions/54250/easy-to-read-java-solution/

**Description:** Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.

**Constraints:** 1 <= nums.length <= 10^5, nums[i] is either 0 or 1.

**Examples:** 
```python3
nums = [0,1] #=> 2
nums = [0,1,0] #=> 2
```

**Hint:** Maybe Recursive DP. Keep track of global max, if curr == 0 subract 1 from count if curr == 1 add 1.  If count == 0, start new subarray length 1. If count is in hashmap replace it iff new count is longer if it is not in the hashmap, add a new hashmap for it

```python3
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        self.reverse(nums, 0, len(nums) - 1)
        self.reverse(nums, 0, k - 1)
        self.reverse(nums, k, len(nums) - 1)
    
    def reverse(self, nums, start, end):
        while start < end:
            temp = nums[start]
            nums[start] = nums[end]
            nums[end] = temp
            start += 1
            end -= 1
```
**Time:** O(n)
**Space:** O(n)

## 19. Subarray Sum Equals K
**Reference:** https://leetcode.com/problems/subarray-sum-equals-k/editorial/

**Description:** Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k. A subarray is a contiguous non-empty sequence of elements within an array.

**Constraints:** 1 <= nums.length <= 2 * 10^4, -1000 <= nums[i] <= 1000, -10^7 <= k <= 10^7

**Examples:** 
```python3
nums = [1,1,1], k = 2 #=> 2
nums = [1,2,3], k = 3] #=> 2
```

**Hint:** Maybe Recursive DP. Keep a global count, use a HashMap. For each number in arr, increament local sum if sum - k is in the hashmap, increment the global count, either way add the sum to the hash map or increment it if it is already there

```python3
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        count, s, res = 0, 0, 0
        map = {0: 1}
        for i in range(len(nums)):
            s += nums[i]
            if s - k in map:
                count += map[s - k]
            map[s] = map.get(s, 0) + 1
        return count
```
**Time:** O(n)
**Space:** O(n)

## 20. Meeting Rooms II
**Reference:** https://aaronice.gitbook.io/lintcode/sweep-line/meeting-rooms-ii
https://github.com/neetcode-gh/leetcode/blob/main/python/0253-meeting-rooms-ii.py

**Description:** Given an array of meeting time intervals consisting of start and end times[[s1,e1],[s2,e2],...](si< ei), find the minimum number of conference rooms required.

**Constraints:** 1 <= nums.length <= 2 * 10^4, -1000 <= nums[i] <= 1000, -10^7 <= k <= 10^7

**Examples:** 
```python3
[[0, 30],[5, 10],[15, 20] #=> 2
[[7,10],[2,4]] #=> 1
```

**Hint:** Split the start and end times into separate arrays. Sort each in ascending order. Consider each meeting start time. If it ends before the current meeting end time, you need a new room. If not check the next meeting end time

```python3
def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        time = []
        for start, end in intervals:
            time.append((start, 1))
            time.append((end, -1))
        
        time.sort(key=lambda x: (x[0], x[1]))
        
        count = 0
        max_count = 0
        for t in time:
            count += t[1]
            max_count = max(max_count, count)
        return max_count
```
**Time:** O((N * logN) + (M * logM))
**Space:** O(1)

## 21. 3Sum Closest
**Reference:** https://leetcode.com/problems/3sum-closest/solutions/7871/python-o-n-2-solution/

**Description:** Given an integer array nums of length n and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

**Constraints:** 3 <= nums.length <= 500, -1000 <= nums[i] <= 1000, -10^4 <= target <= 10^4

**Examples:** 
```python3
nums = [-1,2,1,-4], target = 1 #=> 2
nums = [0,0,0], target = 1  #=> 0
```

**Hint:** Same as 3sum with global max

```python3
class Solution:
    def threeSumClosest(self, num, target):
        num.sort()
        result = num[0] + num[1] + num[2]
        for i in range(len(num) - 2):
            j, k = i+1, len(num) - 1
            while j < k:
                sum = num[i] + num[j] + num[k]
                if sum == target:
                    return sum
                if abs(sum - target) < abs(result - target):
                    result = sum
                if sum < target:
                    j += 1
                elif sum > target:
                    k -= 1
        return result
```
**Time:** O(n^2)
**Space:** O(n)

## 22. Non-overlapping Intervals
**Reference:** https://leetcode.com/problems/non-overlapping-intervals/solutions/276056/python-greedy-interval-scheduling/

**Description:** Given an array of intervals intervals where intervals[i] = [starti, endi], return the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.

**Constraints:** 1 <= intervals.length <= 10^5, intervals[i].length == 2, -5 * 10^4 <= starti < endi <= 5 * 10^4

**Examples:** 
```python3
intervals = [[1,2],[2,3],[3,4],[1,3]] #=> 1
intervals = [[1,2],[1,2],[1,2]]  #=> 2
intervals = [[1,2],[2,3]] #=> 0
```

**Hint:** Sort the list. Iterate over list. If prev end is past curr start, increment the count. Otherwise, prev becomes curr

```python3
def eraseOverlapIntervals(intervals):
	end, cnt = float('-inf'), 0
	for s, e in sorted(intervals, key=lambda x: x[1]):
		if s >= end: 
			end = e
		else: 
			cnt += 1
	return cnt
```
**Time:** O(nlog(n))
**Space:** O(1)

## 23. Employee Free Time
**Reference:** https://aaronice.gitbook.io/lintcode/sweep-line/employee-free-time

**Description:** We are given a list scheduleof employees, which represents the working time for each employee.
Each employee has a list of non-overlappingIntervals, and these intervals are in sorted order.
Return the list of finite intervals representing common, positive-length free time forallemployees, also in sorted order. 

(Even though we are representing Intervals in the form [x, y], the objects inside are Intervals, not lists or arrays. For example, schedule[0][0].start = 1, schedule[0][0].end = 2, and schedule[0][0][0] is not defined.)

Also, we wouldn't include intervals like [5, 5] in our answer, as they have zero length. 0 <= schedule[i].start < schedule[i].end <= 10^8.

**Constraints:** schedule and schedule[i] are lists with lengths in range [1, 50].

**Examples:** 
```python3
schedule = [[[1,2],[5,6]],[[1,3]],[[4,10]]] #=> [[3,4]]
schedule = [[[1,3],[6,7]],[[2,4]],[[2,5],[9,12]]]  #=> [[5,6],[7,9]]
```

**Hint:** Sort the intervals by start times. Initialize temp to be the first interval. Iterate over the interval list.  If temp.end < curr.start (no overlap) add that interval to the output and set temp to current. Otherwise, if there is overlap and the current interval ends after temp set temp to be the current interval.

```java
class Solution {
    public List<Interval> employeeFreeTime(List<List<Interval>> avails) {
        List<Interval> result = new ArrayList<>();
        List<Interval> timeLine = new ArrayList<>();
        avails.forEach(e -> timeLine.addAll(e));
        Collections.sort(timeLine, ((a, b) -> a.start - b.start));

        Interval temp = timeLine.get(0);
        for (Interval each : timeLine) {
            if (temp.end < each.start) {
                result.add(new Interval(temp.end, each.start));
                temp = each;
            } else {
                temp = temp.end < each.end ? each : temp;
            }
        }
        return result;
    }
}
```
**Time:** O(nlogn)
**Space:** O(n)

## 24. Sliding Window Maximum
**Reference:** https://leetcode.com/problems/sliding-window-maximum/

**Description:** You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

**Constraints:** 1 <= nums.length <= 10^5, -10^4 <= nums[i] <= 10^4, 1 <= k <= nums.length

**Examples:** 
```python3
nums = [1,3,-1,-3,5,3,6,7], k = 3 #=> [3,3,5,5,6,7]
nums = [1], k = 1  #=> [1]
```

**Hint:** Use deque. Find the max value in the initial window position and save that value in the output array. Start with nums[0] in queue. If new element is smaller, add it to right. Add left most val to output. Max value is always left most value. Remove left value when it is out of bounds. When adding a new value to queue, remove all smaller values. *Monotonically decreasing queue [queue always decreasing]

```python3
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        output = []
        q = collections.deque()  # index
        l = r = 0
        # O(n) O(n)
        while r < len(nums):
            # pop smaller values from q
            while q and nums[q[-1]] < nums[r]:
                q.pop()
            q.append(r)
            # remove left val from window
            if l > q[0]:
                q.popleft()
            if (r + 1) >= k:
                output.append(nums[q[0]])
                l += 1
            r += 1
        return output
```
**Time:** O(n)
**Space:** O(n)

## 25. Valid Parentheses
**Reference:** https://leetcode.com/problems/valid-parentheses/

**Description:** Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.
3. Every close bracket has a corresponding open bracket of the same type.

**Constraints:** 1 <= s.length <= 10^4, s consists of parentheses only '()[]{}'.

**Examples:** 
```python3
s = "()" #=> true
s = "()[]{}" #=> true
s = "(]" #=> false
```

**Hint:** Create a map that matches right and left brackets. Iterate over input string. If curr is right bracket and matches head of stack, pop stack. If bracket is right and does not match stack head, return false. If left bracket, push to stack. Return true if stack is empty

```python3
class Solution:
    def isValid(self, s: str) -> bool:
        Map = {")": "(", "]": "[", "}": "{"}
        stack = []
        for c in s:
            if c not in Map:
                stack.append(c)
                continue
            if not stack or stack[-1] != Map[c]:
                return False
            stack.pop()
        return not stack
```
**Time:** O(n)
**Space:** O(n)

## 26. Implement Queue using Stacks
**Reference:** https://leetcode.com/problems/implement-queue-using-stacks/solutions/64206/short-o-1-amortized-c-java-ruby/

**Description:** Implement a first in first out (FIFO) queue using only two stacks. The implemented queue should support all the functions of a normal queue (push, peek, pop, and empty).

Implement the MyQueue class:
1. void push(int x) Pushes element x to the back of the queue.
2. int pop() Removes the element from the front of the queue and returns it.
3. int peek() Returns the element at the front of the queue.
4. boolean empty() Returns true if the queue is empty, false otherwise.

Notes:
1. You must use only standard operations of a stack, which means only push to top, peek/pop from top, size, and is empty operations are valid.
2. Depending on your language, the stack may not be supported natively. You may simulate a stack using a list or deque (double-ended queue) as long as you use only a stack's standard operations.

Follow-up: Can you implement the queue such that each operation is amortized O(1) time complexity? In other words, performing n operations will take overall O(n) time even if one of those operations may take longer.

**Constraints:** 1 <= x <= 9, At most 100 calls will be made to push, pop, peek, and empty. All the calls to pop and peek are valid

**Examples:** 
```python3
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []] #=> [null, null, null, 1, 1, false]
```

**Hint:** Use and input stack and and output stack. Push to input stack. Pop for output stack. Empty checks both stacks, and peek returns the end of the output stack after moving all elements to it from input stack 

```python3
class Queue:
    def __init__(self):
        self._in, self._out = [], []

    def push(self, x):
        self._in.append(x)

    def pop(self):
        self.peek()
        return self._out.pop()

    def peek(self):
        if not self._out:
            while self._in:
                self._out.append(self._in.pop())
        return self._out[-1]

    def empty(self):
        return not self._in and not self._out
```
**Time:** O(1)
**Space:** O(1)

## 27. Backspace String Compare
**Reference:** https://leetcode.com/problems/backspace-string-compare/solutions/135603/java-c-python-o-n-time-and-o-1-space/

**Description:** Given two strings s and t, return true if they are equal when both are typed into empty text editors. '#' means a backspace character. Note that after backspacing an empty text, the text will continue empty. Follow up: Can you solve it in O(n) time and O(1) space?

**Constraints:** 1 <= s.length, t.length <= 200, s and t only contain lowercase letters and '#' characters.

**Examples:** 
```python3
s = "ab#c", t = "ad#c" #=> true
s = "ab##", t = "c#d#"  #=> true
s = "a#c", t = "b" #=> false
```

**Hint:** From inside infinite loop. Iterate from end, keep if char, skip next char if you see #

```python3
    def backspaceCompare(self, S, T):
        i, j = len(S) - 1, len(T) - 1
        backS = backT = 0
        while True:
            while i >= 0 and (backS or S[i] == '#'):
                backS += 1 if S[i] == '#' else -1
                i -= 1
            while j >= 0 and (backT or T[j] == '#'):
                backT += 1 if T[j] == '#' else -1
                j -= 1
            if not (i >= 0 and j >= 0 and S[i] == T[j]):
                return i == j == -1
            i, j = i - 1, j - 1
```
**Time:** O(n)
**Space:** O(1)

## 28. Evaluate Reverse Polish Notation
**Reference:** https://leetcode.com/problems/evaluate-reverse-polish-notation/

**Description:** You are given an array of strings tokens that represents an arithmetic expression in a Reverse Polish Notation. Evaluate the expression. Return an integer that represents the value of the expression.

Note that:
1. The valid operators are '+', '-', '*', and '/'.
2. Each operand may be an integer or another expression.
3. The division between two integers always truncates toward zero.
4. There will not be any division by zero.
5. The input represents a valid arithmetic expression in a reverse polish notation.
6. The answer and all the intermediate calculations can be represented in a 32-bit integer.
 

**Constraints:** 1 <= tokens.length <= 10^4, tokens[i] is either an operator: "+", "-", "*", or "/", or an integer in the range [-200, 200].

**Examples:** 
```python3
tokens = ["2","1","+","3","*"] #=> 9
tokens = ["4","13","5","/","+"] #=> 6
```

**Hint:** 

```javascript
const OPERATORS = new Set(["+", "-", "/", "*"]);

const evalRPN = (tokens) => {
    const stack = [];
    for (const token of tokens) {
        if (!OPERATORS.has(token)) {
            stack.push(Number(token));
            continue;
        }
        const number2 = stack.pop();
        const number1 = stack.pop();
        switch (token) {
            case "+": {
                stack.push(number1 + number2);
                break;
            } case "-": {
                stack.push(number1 - number2);
                break;
            } case "/": {
                stack.push(Math.trunc(number1 / number2));
                break;
            } case "*": {
                stack.push(number1 * number2);
                break;
            }
        }
    }
    return stack.pop();
};
```
**Time:** O(n)
**Space:** O(n)

## 29. Min Stack
**Reference:** https://leetcode.com/problems/min-stack/solutions/514932/min-stack/

**Description:** Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.

Implement the MinStack class:

1. MinStack() initializes the stack object.
2. void push(int val) pushes the element val onto the stack.
3. void pop() removes the element on the top of the stack.
4. int top() gets the top element of the stack.
5. int getMin() retrieves the minimum element in the stack.
You must implement a solution with O(1) time complexity for each function.

**Constraints:** -2^31 <= val <= 2^31 - 1, Methods pop, top and getMin operations will always be called on non-empty stacks. At most 3 * 10^4 calls will be made to push, pop, top, and getMin.

**Examples:** 
```python3
["MinStack","push","push","push","getMin","pop","top","getMin"]
[[],[-2],[0],[-3],[],[],[],[]] #=> [null,null,null,null,-3,null,0,-2]
```

**Hint:** Create two stacks. Push values to one stack. Push current min and its count to the other stack (i.e., tuple). Pop the min tracker stack when the tuple count becomes zero

```javascript
function last(arr) {
    return arr[arr.length - 1];
}

class MinStack {
    _stack = [];
    _minStack = [];

    push(x) {
        // We always put the number onto the main stack.
        this._stack.push(x);

        // If the min stack is empty, or this number is smaller
        // than the top of the min stack, put it on with a count of 1.
        if (this._minStack.length === 0 || x < last(this._minStack)[0]) {
            this._minStack.push([x, 1]);
        }
        // Else if this number is equal to what's currently at the top
        // of the min stack, then increment the count at the top by 1.
        else if (x === last(this._minStack)[0]) {
            last(this._minStack)[1]++;
        }
    }

    pop() {
        // If the top of min stack is the same as the top of stack
        // then we need to decrement the count at the top by 1.
        if (last(this._minStack)[0] === last(this._stack)) {
            last(this._minStack)[1]--;
        }

        // If the count at the top of min stack is now 0, then remove
        // that value as we're done with it.
        if (last(this._minStack)[1] === 0) {
            this._minStack.pop();
        }

        // And like before, pop the top of the main stack.
        this._stack.pop();
    }

    top() {
        return last(this._stack);
    }

    getMin() {
        return last(this._minStack)[0];
    }
}
```
**Time:** O(1)
**Space:** O(n)

## 30. Daily Temperatures
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0739-daily-temperatures.py

**Description:** Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.

**Constraints:** 1 <= temperatures.length <= 10^5, 30 <= temperatures[i] <= 100

**Examples:** 
```python3
temperatures = [73,74,75,71,69,72,76,73] #=> [1,1,4,2,1,1,0,0]
temperatures = [30,40,50,60] #=> [1,1,1,0]
temperatures = [30,60,90] #=> [1,1,0]
```

**Hint:** Create an answer array with same length as temperature array and initialize it to zeros. Iterate over temperatures. If stack is not empty, there are previous days that have not seen a warmer day.  While curr temp is > preday, set answer[prevDay] = currDay - prevDay. Push curr idx (currDay) onto stack. Note: you are pushing an offset, temp tuple. Return stack

```python3
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        res = [0] * len(temperatures)
        stack = []  # pair: [temp, index]

        for i, t in enumerate(temperatures):
            while stack and t > stack[-1][0]:
                stackT, stackInd = stack.pop()
                res[stackInd] = i - stackInd
            stack.append((t, i))
        return res
```
**Time:** O(n)
**Space:** O(n) note: (O(1) space possible)

## 31. Decode String
**Reference:** https://leetcode.com/problems/decode-string/solutions/941309/python-stack-solution-explained/

**Description:** Given an encoded string, return its decoded string. The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer. You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].
The test cases are generated so that the length of the output will never exceed 10^5.

**Constraints:** 1 <= s.length <= 30, s consists of lowercase English letters, digits, and square brackets '[]', s is guaranteed to be a valid input. All the integers in s are in the range [1, 300].

**Examples:** 
```python3
s = "3[a]2[bc]" #=> "aaabcbc"
s = "3[a2[c]]" #=> "accaccacc"
s = "2[abc]3[cd]ef" #=> "abcabccdcdcdef"
```

**Hint:** Maybe use recursion? If we see digit, it means that we need to form number, so just do it: multiply already formed number by 10 and add this digit.
If we see open bracket [, it means, that we just right before finished to form our number: so we put it into our stack. Also we put in our stack empty string.
If we have close bracket ], it means that we just finished [...] block and what we have in our stack: on the top it is solution for what we have inside bracktes, before we have number of repetitions of this string rep and finally, before we have string built previously: so we concatenate str2 and str1 * rep.
Finally, if we have some other symbol, that is letter, we add it the the last element of our stack.

```python3
class Solution:
    def decodeString(self, s):
        it, num, stack = 0, 0, [""]
        while it < len(s):
            if s[it].isdigit():
                num = num * 10 + int(s[it])
            elif s[it] == "[":
                stack.append(num)
                num = 0
                stack.append("")
            elif s[it] == "]":
                str1 = stack.pop()
                rep = stack.pop()
                str2 = stack.pop()
                stack.append(str2 + str1 * rep)
            else:
                stack[-1] += s[it]              
            it += 1           
        return "".join(stack)
```
**Time:** O(n)
**Space:** O(n)

## 32. Asteroid Collision
**Reference:** https://leetcode.com/problems/asteroid-collision/solutions/193403/java-easy-to-understand-solution/

**Description:** We are given an array asteroids of integers representing asteroids in a row. For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed. Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

**Constraints:** 2 <= asteroids.length <= 10^4, -1000 <= asteroids[i] <= 1000, asteroids[i] != 0

**Examples:** 
```python3
asteroids = [5,10,-5] #=> [5,10]
asteroids = [8,-8] #=> []
asteroids = [10,2,-5] #=> [10]
```

**Hint:** If the asteroid is with + sign, simply push onto stack since it can't collide, irrespective of whether the stack top is + (both same direction & hence can't collide) or stack top is - (since both in opposite direction & the stack top is present to left of the asteroid & also moving left, they can't collide)

If the asteroid is with - sign, there can be couple of cases :
1. if stack top is +ve & absolute value is lesser than the asteroid, then it has to be blown off, so pop it off.
2. if the stack top is also -ve, simply push the asteroid, no question of collision since both move in left direction.
3. if the absolute value of asteroid & stack top are same, both would be blown off, so effectively pop off from stack & do nothing with the current asteroid.

```python3
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        s = []
        for i in asteroids:
            if i > 0:
                s.append(i)
            else:  # i is negative
                while s and s[-1] > 0 and s[-1] < abs(i):
                    s.pop()
                if not s or s[-1] < 0:
                    s.append(i)
                elif i + s[-1] == 0:
                    s.pop()  # equal
        return s
```
**Time:** O(n)
**Space:** O(n)

## 33. Basic Calculator II
**Reference:** https://leetcode.com/problems/basic-calculator-ii/solutions/63003/share-my-java-solution/

**Description:** Given a string s which represents an expression, evaluate this expression and return its value. The integer division should truncate toward zero. You may assume that the given expression is always valid. All intermediate results will be in the range of [-2^31, 2^31 - 1]. Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

**Constraints:**
1. 1 <= s.length <= 3 * 10^5
2. s consists of integers and operators ('+', '-', '*', '/') separated by some number of spaces.
3. s represents a valid expression.
4. All the integers in the expression are non-negative integers in the range [0, 2^31 - 1].
5. The answer is guaranteed to fit in a 32-bit integer.

**Examples:** 
```python3
s = "3+2*2" #=> 7
s = " 3/2 " #=> 1
s = " 3+5 / 2 " #=> 5
```

**Hint:** Iterate over input string. If you encounter a digit append it to any adjacent digits that you just saw in num variable. If + operator, append num to stack. If - operator append -num to stack. If * or / pop stack and * or / num and popped then push result.  At the end of the loop, the stack will contain only numbers. Add all of them together and return the result

```python
class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        num = 0
        sign = '+'
        for idx in range(len(s)):
            if s[idx].isdigit():
                num = num * 10 + int(s[idx])
            if not s[idx].isdigit() and s[idx] != ' ' or idx == len(s) - 1:
                if sign == '-': stack.append(-num)
                if sign == '+': stack.append(num)
                if sign == '*': stack.append(stack.pop()*num)
                if sign == '/':  stack.append(int(stack.pop()/num))
                sign = s[idx]
                num = 0
        return sum(stack) 
```
**Time:** O(n)
**Space:** O(n)

## 34. Trapping Rain Water
**Reference:** https://leetcode.com/problems/trapping-rain-water/solutions/17357/sharing-my-simple-c-code-o-n-time-o-1-space/

**Description:** Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.

![image](https://github.com/will4skill/algo-review/assets/10373005/3c2ddbed-c3e1-41bd-8d2c-5a7c8e678f50)

**Constraints:** n == height.length,  1 <= n <= 2 * 10^4, 0 <= height[i] <= 10^5

**Examples:** 
```python3
height = [0,1,0,2,1,0,1,3,2,1,2,1] #=> 6
height = [4,2,0,3,2,5] #=> 9
```

**Hint:** Note: Move smaller height's pointer toward middle. Create left and right pointers at ends of array. Iterate until they converge. If left height < right height if left height >= leftMax, update leftMax otherwise, increment answer with leftMax - height[left]. Either way, increment left pointer. If left height >= right height repeat proces on right side, but decrement right pointer

```python3
class Solution:
    def trap(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        res = 0
        maxleft, maxright = 0, 0
        while left <= right:
            if height[left] <= height[right]:
                if height[left] >= maxleft:
                    maxleft = height[left]
                else:
                    res += maxleft - height[left]
                left += 1
            else:
                if height[right] >= maxright:
                    maxright = height[right]
                else:
                    res += maxright - height[right]
                right -= 1
        return res
```
**Time:** O(n)
**Space:** O(1)

## 35. Basic Calculator
**Reference:** https://leetcode.com/problems/basic-calculator/solutions/1456850/python-basic-calculator-i-ii-iii-easy-solution-detailed-explanation/

**Description:** Given a string s representing a valid expression, implement a basic calculator to evaluate it, and return the result of the evaluation. Note: You are not allowed to use any built-in function which evaluates strings as mathematical expressions, such as eval().

**Constraints:**
1 <= s.length <= 3 * 10^5
s consists of digits, '+', '-', '(', ')', and ' '.
s represents a valid expression.
'+' is not used as a unary operation (i.e., "+1" and "+(2 + 3)" is invalid).
'-' could be used as a unary operation (i.e., "-1" and "-(2 + 3)" is valid).
There will be no two consecutive operators in the input.
Every number and running calculation will fit in a signed 32-bit integer.

**Examples:** 
```python3
s = "1 + 1" #=> 2
s = " 2-1 + 2 " #=> 3
s = "(1+(4+5+2)-3)+(6+8)" #=> 23
```

**Hint:** Iterate over input string. If you encounter a digit append it to any adjacent digits that you just saw in num variable. If + operator, append num to stack. If - operator append -num to stack. If * or / pop stack and * or / num and popped then push result. If you encounter a "(" you recurse with the rest of the string. If you encounter a ")" you return from a recursion. At the end of the loop, the stack will contain only numbers. Add all of them together and return the result

```python3
class Solution:
    def calculate(self, s):    
        def calc(it):
            def update(op, v):
                if op == "+": stack.append(v)
                if op == "-": stack.append(-v)
                if op == "*": stack.append(stack.pop() * v)
                if op == "/": stack.append(int(stack.pop() / v))
        
            num, stack, sign = 0, [], "+"
            
            while it < len(s):
                if s[it].isdigit():
                    num = num * 10 + int(s[it])
                elif s[it] in "+-*/":
                    update(sign, num)
                    num, sign = 0, s[it]
                elif s[it] == "(":
                    num, j = calc(it + 1)
                    it = j - 1
                elif s[it] == ")":
                    update(sign, num)
                    return sum(stack), it + 1
                it += 1
            update(sign, num)
            return sum(stack)

        return calc(0)
```
**Time:** O(n)
**Space:** O(n)

## 36. Largest Rectangle in Histogram
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0084-largest-rectangle-in-histogram.py

**Description:** Given an array of integers heights representing the histogram's bar height where the width of each bar is 1, return the area of the largest rectangle in the histogram.

**Constraints:** 1 <= heights.length <= 10^5, 0 <= heights[i] <= 10^4

**Examples:** 

```python3
heights = [2,1,5,6,2,3] #=> 10
```

![image](https://github.com/will4skill/algo-review/assets/10373005/261a3590-cbd7-4aac-8e44-acd2253b9986)

```python3
heights = [2,4] #=> 4
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d9fb025a-1154-4ac1-9ae5-9431d069ef2b)


**Hint:** For each height, store height and start idx (how far back you can extend) tuple in stack. Add new heights until the new height is shorter than the prev top of stack. Compute the previous max area. Pop the stack until you encounter a smaller previous than the new height. Iterate through the remaining stack elements and compute the max area between those and the main loop max area.

```python3
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        maxArea = 0
        stack = []  # pair: (index, height)

        for i, h in enumerate(heights):
            start = i
            while stack and stack[-1][1] > h:
                index, height = stack.pop()
                maxArea = max(maxArea, height * (i - index))
                start = index
            stack.append((start, h))

        for i, h in stack:
            maxArea = max(maxArea, h * (len(heights) - i))
        return maxArea
```
**Time:** O(n)
**Space:** O(n) 

## 37. Maximum Frequency Stack
**Reference:** https://leetcode.com/problems/maximum-frequency-stack/solutions/163410/c-java-python-o-1/

**Description:** Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the FreqStack class:
1. FreqStack() constructs an empty frequency stack.
2. void push(int val) pushes an integer val onto the top of the stack.
3. int pop() removes and returns the most frequent element in the stack.
   a. If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.

**Constraints:** 
0 <= val <= 10^9
At most 2 * 10^4 calls will be made to push and pop.
It is guaranteed that there will be at least one element in the stack before calling pop.

**Examples:** 
```python3
["FreqStack", "push", "push", "push", "push", "push", "push", "pop", "pop", "pop", "pop"]
[[], [5], [7], [5], [7], [4], [5], [], [], [], []] #=> [null, null, null, null, null, null, null, 5, 7, 5, 4]
```

**Hint:** Use a freq to count the freq of elements. m is a map of stack. If element x has n frequence, we will push x n times in m[1], m[2] .. m[n]
maxfreq records the maximum frequence. 
push(x) will push x to m[++freq[x]]
pop() will pop from the m[maxfreq]

```python3
    def __init__(self):
        self.freq = collections.Counter()
        self.m = collections.defaultdict(list)
        self.maxf = 0

    def push(self, x):
        freq, m = self.freq, self.m
        freq[x] += 1
        self.maxf = max(self.maxf, freq[x])
        m[freq[x]].append(x)

    def pop(self):
        freq, m, maxf = self.freq, self.m, self.maxf
        x = m[maxf].pop()
        if not m[maxf]: self.maxf = maxf - 1
        freq[x] -= 1
        return x
```
**Time:** O(1)
**Space:** O(n)

## 38. Longest Valid Parentheses
**Reference:** https://leetcode.com/problems/longest-valid-parentheses/solutions/14126/my-o-n-solution-using-a-stack/

**Description:** Given a string containing just the characters '(' and ')', return the length of the longest valid (well-formed) parentheses 

**Constraints:** 0 <= s.length <= 3 * 10^4, s[i] is '(', or ')'.

**Examples:** 
```python3
s = "(()" #=> 2
s = ")()())" #=> 4
s = "" #=> 0
```

**Hint:** Scan the string from beginning to end.
If current character is '(',
push its index to the stack. If current character is ')' and the
character at the index of the top of stack is '(', we just find a
matching pair so pop from the stack. Otherwise, we push the index of
')' to the stack.
After the scan is done, the stack will only
contain the indices of characters which cannot be matched. 
If the stack is empty, the whole input
string is valid. Otherwise, we can scan the stack to get longest
valid substring: use the opposite side - substring between adjacent indices
should be valid parentheses.

```python3
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        n = len(s)
        longest = 0
        st = []
        for i in range(n):
            if s[i] == '(':
                st.append(i)
            else:
                if st and s[st[-1]] == '(':
                    st.pop()
                else:
                    st.append(i)
        if not st:
            longest = n
        else:
            a, b = n, 0
            while st:
                b = st[-1]
                st.pop()
                longest = max(longest, a - b - 1)
                a = b
            longest = max(longest, a)
        return longest
```
**Time:** O(n)
**Space:** O(n)

## 39. Merge Two Sorted Lists
**Reference:** https://leetcode.com/problems/merge-two-sorted-lists/solutions/1826666/c-easy-to-understand-2-approaches-recursive-iterative/

**Description:** You are given the heads of two sorted linked lists list1 and list2. Merge the two lists into one sorted list. The list should be made by splicing together the nodes of the first two lists.Return the head of the merged linked list.

**Constraints:**
The number of nodes in both lists is in the range [0, 50].
-100 <= Node.val <= 100
Both list1 and list2 are sorted in non-decreasing order.

**Examples:** 
```python3
list1 = [1,2,4], list2 = [1,3,4] #=> [1,1,2,3,4,4]
list1 = [], list2 = [] #=> []
list1 = [], list2 = [0] #=> [0]
```

**Hint:** Use the merge logic from merge sort. Don't forget the longer list leftovers at end. Note: You can also use recursion if you want

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
       
        // if list1 happen to be NULL
        // we will simply return list2.
        if(list1 == NULL)
            return list2;
		
        // if list2 happen to be NULL
        // we will simply return list1.
        if(list2 == NULL)
            return list1;
        
        ListNode * ptr = list1;
        if(list1 -> val > list2 -> val)
        {
            ptr = list2;
            list2 = list2 -> next;
        }
        else
        {
            list1 = list1 -> next;
        }
        ListNode *curr = ptr;
        
        // till one of the list doesn't reaches NULL
        while(list1 &&  list2)
        {
            if(list1 -> val < list2 -> val){
                curr->next = list1;
                list1 = list1 -> next;
            }
            else{
                curr->next = list2;
                list2 = list2 -> next;
            }
            curr = curr -> next;
                
        }
		
        // adding remaining elements of bigger list.
        if(!list1)
            curr -> next = list2;
        else
            curr -> next = list1;
            
        return ptr;
       
    }
};
```
**Time:** O(n)
**Space:** O(1)

## 40. Linked List Cycle
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0739-daily-temperatures.py

**Description:** Given head, the head of a linked list, determine if the linked list has a cycle in it. There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter. Return true if there is a cycle in the linked list. Otherwise, return false.

Follow up: Can you solve it using O(1) (i.e. constant) memory?

**Constraints:** 
The number of the nodes in the list is in the range [0, 10^4].
-10^5 <= Node.val <= 10^5
pos is -1 or a valid index in the linked-list.

**Examples:** 

```python3
head = [3,2,0,-4], pos = 1 #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/6cbc219c-8680-4a93-a9e6-8e660775555c)

```python3
head = [1,2], pos = 0 #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/4895e03b-0ae6-46c9-b64a-3e1db97b2959)

```python3
head = [1], pos = -1 #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/40f53512-147f-40a5-89ff-d2cc4d13efe3)

**Hint:** Use a fast (2x) pointer and slow pointer, if they converge before reaching the end there is a cycle. 

```python3
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if head == None: return False
        slowPtr = head
        fastPtr = head.next
        while fastPtr != None and fastPtr.next != None:
            if slowPtr == fastPtr: return True
            slowPtr = slowPtr.next
            fastPtr = fastPtr.next.next
        return False
```
**Time:** O(n)
**Space:** O(1)

## 41. Reverse Linked List
**Reference:** https://www.structy.net/problems/reverse-list

**Description:** Given the head of a singly linked list, reverse the list, and return the reversed list. Follow up: A linked list can be reversed either iteratively or recursively. Could you implement both?

**Constraints:** The number of nodes in the list is the range [0, 5000]. -5000 <= Node.val <= 5000

**Examples:** 
```python3
head = [1,2,3,4,5] #=> [5,4,3,2,1]
head = [1,2] #=> [2,1]
head = [] #=> []
```

**Hint:** You need ptrs to curr prev and next

```python3
def reverse_list(head):
  prev = None
  current = head
  while current is not None:
    next = current.next
    current.next = prev
    prev = current
    current = next
  return prev
```
**Time:** O(n)
**Space:** O(1)

## 42. Middle of the Linked List
**Reference:** https://leetcode.com/problems/middle-of-the-linked-list/solutions/154619/c-java-python-slow-and-fast-pointers/

**Description:** Given the head of a singly linked list, return the middle node of the linked list. If there are two middle nodes, return the second middle node.

**Constraints:** The number of nodes in the list is in the range [1, 100]. 1 <= Node.val <= 100

**Examples:** 
```python3
head = [1,2,3,4,5] #=> [3,4,5]
head = [1,2,3,4,5,6] #=> [4,5,6]
```

**Hint:** Fast ptr (2x), slow ptr (1x)

```python3
    def middleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```
**Time:** O(n)
**Space:** O(1)

## 43. Palindrome Linked List
**Reference:** https://leetcode.com/problems/palindrome-linked-list/solutions/64501/java-easy-to-understand/

**Description:** Given the head of a singly linked list, return true if it is a palindrome or false otherwise. Follow up: Could you do it in O(n) time and O(1) space?

**Constraints:** The number of nodes in the list is in the range [1, 10^5]. 0 <= Node.val <= 9

**Examples:** 
```python3
head = [1,2,2,1] #=> true
head = [1,2] #=> false
```

**Hint:** Use fast and slow to get ptrs to the mid and end. Reverse from end to mid, iterate toward middle comparing the chars

```java
public boolean isPalindrome(ListNode head) {
    ListNode fast = head, slow = head;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        slow = slow.next;
    }
    if (fast != null) { // odd nodes: let right half smaller
        slow = slow.next;
    }
    slow = reverse(slow);
    fast = head;
    
    while (slow != null) {
        if (fast.val != slow.val) {
            return false;
        }
        fast = fast.next;
        slow = slow.next;
    }
    return true;
}

public ListNode reverse(ListNode head) {
    ListNode prev = null;
    while (head != null) {
        ListNode next = head.next;
        head.next = prev;
        prev = head;
        head = next;
    }
    return prev;
}
```
**Time:** O(n)
**Space:** O(1)

## 44. LRU Cache
**Reference:** https://leetcode.com/problems/lru-cache/solutions/45911/java-hashtable-double-linked-list-with-a-touch-of-pseudo-nodes/

**Description:** Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:
1. LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
2. int get(int key) Return the value of the key if the key exists, otherwise return -1.
3. void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
The functions get and put must each run in O(1) average time complexity.

**Constraints:** 
1 <= capacity <= 3000
0 <= key <= 10^4
0 <= value <= 10^5
At most 2 * 10^5 calls will be made to get and put.

**Examples:** 
```python3
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]] #=> [null, null, null, 1, null, -1, null, -1, 3, 4]
```

**Hint:** You can use a hashTable with integer keys and double linked list values

```java
import java.util.Hashtable;

public class LRUCache {

class DLinkedNode {
  int key;
  int value;
  DLinkedNode pre;
  DLinkedNode post;
}

/**
 * Always add the new node right after head;
 */
private void addNode(DLinkedNode node) {
    
  node.pre = head;
  node.post = head.post;

  head.post.pre = node;
  head.post = node;
}

/**
 * Remove an existing node from the linked list.
 */
private void removeNode(DLinkedNode node){
  DLinkedNode pre = node.pre;
  DLinkedNode post = node.post;

  pre.post = post;
  post.pre = pre;
}

/**
 * Move certain node in between to the head.
 */
private void moveToHead(DLinkedNode node){
  this.removeNode(node);
  this.addNode(node);
}

// pop the current tail. 
private DLinkedNode popTail(){
  DLinkedNode res = tail.pre;
  this.removeNode(res);
  return res;
}

private Hashtable<Integer, DLinkedNode> 
  cache = new Hashtable<Integer, DLinkedNode>();
private int count;
private int capacity;
private DLinkedNode head, tail;

public LRUCache(int capacity) {
  this.count = 0;
  this.capacity = capacity;

  head = new DLinkedNode();
  head.pre = null;

  tail = new DLinkedNode();
  tail.post = null;

  head.post = tail;
  tail.pre = head;
}

public int get(int key) {

  DLinkedNode node = cache.get(key);
  if(node == null){
    return -1; // should raise exception here.
  }

  // move the accessed node to the head;
  this.moveToHead(node);

  return node.value;
}


public void put(int key, int value) {
  DLinkedNode node = cache.get(key);

  if(node == null){

    DLinkedNode newNode = new DLinkedNode();
    newNode.key = key;
    newNode.value = value;

    this.cache.put(key, newNode);
    this.addNode(newNode);

    ++count;

    if(count > capacity){
      // pop the tail
      DLinkedNode tail = this.popTail();
      this.cache.remove(tail.key);
      --count;
    }
  }else{
    // update the value.
    node.value = value;
    this.moveToHead(node);
  }
}

}
```
**Time:** O(1)??
**Space:** O(n)

## 45. Remove Nth Node From End of List
**Reference:** https://leetcode.com/problems/remove-nth-node-from-end-of-list/solutions/1164542/js-python-java-c-easy-two-pointer-solution-w-explanation/

**Description:** Given the head of a linked list, remove the nth node from the end of the list and return its head.

**Constraints:** 
The number of nodes in the list is sz.
1 <= sz <= 30
0 <= Node.val <= 100
1 <= n <= sz

**Examples:** 
```python3
head = [1,2,3,4,5], n = 2 #=> [1,2,3,5]
head = [1], n = 1 #=> []
head = [1,2], n = 1 #=> [1]
```

**Hint:** Use fast and slow pointers: "stagger our two pointers by n nodes by giving the first pointer (fast) a head start before starting the second pointer (slow). Doing this will cause slow to reach the n'th node from the end at the same time that fast reaches the end."

```python3
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        fast, slow = head, head
        for _ in range(n): fast = fast.next
        if not fast: return head.next
        while fast.next: fast, slow = fast.next, slow.next
        slow.next = slow.next.next
        return head
```
**Time:** O(n)
**Space:** O(1)

## 46. Swap Nodes in Pairs
**Reference:** https://leetcode.com/problems/swap-nodes-in-pairs/submissions/725221815/

**Description:** Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

**Constraints:** 
The number of nodes in the list is in the range [0, 100].
0 <= Node.val <= 100

**Examples:** 
```python3
head = [1,2,3,4] #=> [2,1,4,3]
head = [] #=> []
head = [1] #=> [1]
```

**Hint:** No real trick, just have to keep track of pointers while iterating over list

```python3
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next:
            return head

        newHead = head.next
        prev = None

        while head and head.next:
            sec = head.next
            third = head.next.next

            head.next = third
            sec.next = head

            if prev:
                prev.next = sec

            prev = head
            head = third

        return newHead
```
**Time:** O(n)
**Space:** O(1)

## 47. Odd Even Linked List
**Reference:** https://leetcode.com/problems/swap-nodes-in-pairs/submissions/725221815/

**Description:** No real trick, just have to keep track of pointers while iterating over list

**Constraints:** 
The number of nodes in the list is in the range [0, 100].
0 <= Node.val <= 100

**Examples:** 
```python3
head = [1,2,3,4] #=> [2,1,4,3]
head = [] #=> []
head = [1] #=> [1]
```

**Hint:** No real trick, just have to keep track of pointers while iterating over list

```python3
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return head
        
        curr_odd = head
        first_even = head.next
        last_even = head.next
        
        while last_even is not None and last_even.next is not None:
            curr_odd.next = last_even.next # point last odd in chain to next odd
            curr_odd = curr_odd.next # increment curr odd pointer
            last_even.next = curr_odd.next # point last even to next node
            last_even = last_even.next # increment last even
            curr_odd.next = first_even # finish chain
        
        return head
```
**Time:** O(n)
**Space:** O(1)

## 48. Add Two Numbers
**Reference:** https://leetcode.com/problems/add-two-numbers/solutions/3675747/beats-100-c-java-python-beginner-friendly/

**Description:** You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list. You may assume the two numbers do not contain any leading zero, except the number 0 itself.

**Constraints:** 
The number of nodes in each linked list is in the range [1, 100].
0 <= Node.val <= 9
It is guaranteed that the list represents a number that does not have leading zeros.

**Examples:** 
```python3
l1 = [2,4,3], l2 = [5,6,4] #=> [7,0,8]
l1 = [0], l2 = [0] #=> [0]
l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9] #=> [8,9,9,9,0,0,0,1]
```

**Hint:** Iterate through both lists, while keeping track of carry

```python3
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        dummyHead = ListNode(0)
        tail = dummyHead
        carry = 0

        while l1 is not None or l2 is not None or carry != 0:
            digit1 = l1.val if l1 is not None else 0
            digit2 = l2.val if l2 is not None else 0

            sum = digit1 + digit2 + carry
            digit = sum % 10
            carry = sum // 10

            newNode = ListNode(digit)
            tail.next = newNode
            tail = tail.next

            l1 = l1.next if l1 is not None else None
            l2 = l2.next if l2 is not None else None

        result = dummyHead.next
        dummyHead.next = None
        return result
```
**Time:** O(n)
**Space:** O(1)

## 49. Sort List
**Reference:** https://leetcode.com/problems/sort-list/solutions/1795126/c-merge-sort-2-pointer-easy-to-understand/

**Description:** Given the head of a linked list, return the list after sorting it in ascending order. Follow up: Can you sort the linked list in O(n logn) time and O(1) memory (i.e. constant space)?

**Constraints:** 
The number of nodes in the list is in the range [0, 5 * 10^4].
-10^5 <= Node.val <= 10^5

**Examples:** 
```python3
head = [4,2,1,3] #=> [1,2,3,4]
head = [-1,5,3,4,0] #=> [-1,0,3,4,5]
head = [] #=> []
```

**Hint:** 
"1. Using 2pointer / fast-slow pointer find the middle node of the list.
2. Now call mergeSort for 2 halves.
3. Merge the Sort List (divide and conqueror Approach)"

```python3
class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head

        temp = None
        slow = head
        fast = head

        while fast is not None and fast.next is not None:
            temp = slow
            slow = slow.next
            fast = fast.next.next

        temp.next = None

        l1 = self.sortList(head)
        l2 = self.sortList(slow)

        return self.mergeList(l1, l2)

    def mergeList(self, l1, l2):
        ptr = ListNode(0)
        curr = ptr

        while l1 is not None and l2 is not None:
            if l1.val <= l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next

            curr = curr.next

        if l1 is not None:
            curr.next = l1

        if l2 is not None:
            curr.next = l2

        return ptr.next
```
**Time:** O(nlog(n)
**Space:** O(1)

## 50. Reorder List
**Reference:** https://leetcode.com/problems/reorder-list/solutions/801883/python-3-steps-to-success-explained/

**Description:** You are given the head of a singly linked-list. The list can be represented as:
L0 → L1 → … → Ln - 1 → Ln
Reorder the list to be on the following form:

L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
You may not modify the values in the list's nodes. Only nodes themselves may be changed.

**Constraints:** 
The number of nodes in the list is in the range [1, 5 * 10^4].
1 <= Node.val <= 1000

**Examples:** 
```python3
head = [1,2,3,4] #=> [1,4,2,3]
head = [1,2,3,4,5] #=> [1,5,2,4,3]
```

**Hint:** Find the middle of the list. Reverse the second half of the list.  Merge the two lists

```python3
class Solution:
    def reorderList(self, head):
        #step 1: find middle
        if not head: return []
        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        
        #step 2: reverse second half
        prev, curr = None, slow.next
        while curr:
            nextt = curr.next
            curr.next = prev
            prev = curr
            curr = nextt    
        slow.next = None
        
        #step 3: merge lists
        head1, head2 = head, prev
        while head2:
            nextt = head1.next
            head1.next = head2
            head1 = head2
            head2 = nextt
```
**Time:** O(n)
**Space:** O(1)

## 51. Rotate List (right by k places)
**Reference:** https://leetcode.com/problems/rotate-list/solutions/22715/share-my-java-solution-with-explanation/

**Description:** Given the head of a linked list, rotate the list to the right by k places.

**Constraints:** 
The number of nodes in the list is in the range [0, 500].
-100 <= Node.val <= 100
0 <= k <= 2 * 10^9

**Examples:** 

```python3
head = [1,2,3,4,5], k = 2 #=> [4,5,1,2,3]
```
![image](https://github.com/will4skill/algo-review/assets/10373005/123dc358-5cad-4bfc-b8a6-9fea4f5c5644)

```python3
head = [0,1,2], k = 4 #=> [2,0,1]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/148d7bc1-cc2b-452d-8b31-759e55eafa6f)

**Hint:** "Since n may be a large number compared to the length of list. So we need to know the length of linked list.After that, move the list after the (l-n%l )th node to the front to finish the rotation."

```python3
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head or not head.next:
            return head
        dummy = ListNode(0)
        dummy.next = head
        fast, slow = dummy, dummy

        i = 0
        while fast.next: # Get the total length 
            fast = fast.next
            i += 1

        for j in range(i - k % i): # Get the i-n%i th node
            slow = slow.next

        fast.next = dummy.next # Do the rotation
        dummy.next = slow.next
        slow.next = None

        return dummy.next
```
**Time:** O(n)
**Space:** O(1)

## 52. Reverse Nodes in k-Group
**Reference:** https://leetcode.com/problems/reverse-nodes-in-k-group/solutions/11440/non-recursive-java-solution-and-idea/

**Description:** Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list. k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
You may not alter the values in the list's nodes, only nodes themselves may be changed.

Follow-up: Can you solve the problem in O(1) extra memory space?

**Constraints:** 
The number of nodes in the list is n.
1 <= k <= n <= 5000
0 <= Node.val <= 1000

**Examples:** 

```python3
head = [1,2,3,4,5], k = 2 #=> [2,1,4,3,5]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/a9db01cf-f2c3-4aa4-aa5a-6291db266280)

```python3
head = [1,2,3,4,5], k = 3 #=> [3,2,1,4,5]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/b65959b0-2523-4a98-a50d-f96cf4d98882)


**Hint:** "First, build a function reverse() to reverse the ListNode between begin and end. Then walk thru the linked list and apply reverse() iteratively. See the code below."

```python3
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        def reverse(begin, end):
            curr = begin.next
            prev = begin
            first = curr
            for _ in range(k):
                if curr:
                    next = curr.next
                    curr.next = prev
                    prev = curr
                    curr = next
            begin.next = prev
            first.next = curr
            return first

        if not head or not head.next or k == 1:
            return head

        dummyhead = ListNode(-1)
        dummyhead.next = head
        begin = dummyhead
        i = 0
        while head:
            i += 1
            if i % k == 0:
                begin = reverse(begin, head.next)
                head = begin.next
            else:
                head = head.next
        return dummyhead.next
```
**Time:** O(n^2) ??
**Space:** O(1)

## 53. Valid Palindrome
**Reference:** https://leetcode.com/problems/valid-palindrome/solutions/350929/solution-in-python-3-beats-100-two-lines-o-1-solution-as-well/

**Description:** A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers. Given a string s, return true if it is a palindrome, or false otherwise.

**Constraints:** 
1 <= s.length <= 2 * 10^5
s consists only of printable ASCII characters.

**Examples:** 
```python3
s = "A man, a plan, a canal: Panama" #=> true
s = "race a car" #=> false
s = " " #=> true
```

**Hint:** Start and end pointers, keep going until they converge

```python3
class Solution:
    def isPalindrome(self, s: str) -> bool:
    	i, j = 0, len(s) - 1
    	while i < j:
    		a, b = s[i].lower(), s[j].lower()
    		if a.isalnum() and b.isalnum():
    			if a != b: return False
    			else:
    				i, j = i + 1, j - 1
    				continue
    		i, j = i + (not a.isalnum()), j - (not b.isalnum())
    	return True
```
**Time:** O(n)
**Space:** O(1)

## 54. Valid Anagram
**Reference:** https://leetcode.com/problems/valid-anagram/solutions/3687854/3-method-s-c-java-python-beginner-friendly/

**Description:** Given two strings s and t, return true if t is an anagram of s, and false otherwise. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once. Follow up: What if the inputs contain Unicode characters? How would you adapt your solution to such a case?

**Constraints:** 
1 <= s.length, t.length <= 5 * 10^4
s and t consist of lowercase English letters.

**Examples:** 
```python3
s = "anagram", t = "nagaram" #=> true
s = "rat", t = "car" #=> false
```

**Hint:** Either sort and compare or use a hashMap to compare character frequencies

```python3
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        count = [0] * 26
        
        # Count the frequency of characters in string s
        for x in s:
            count[ord(x) - ord('a')] += 1
        
        # Decrement the frequency of characters in string t
        for x in t:
            count[ord(x) - ord('a')] -= 1
        
        # Check if any character has non-zero frequency
        for val in count:
            if val != 0:
                return False

        return True
```
**Time:** O(n)
**Space:** O(n)

## 55. Longest Palindrome
**Reference:** https://leetcode.com/problems/longest-palindrome/submissions/1136019451/

**Description:** Given a string s which consists of lowercase or uppercase letters, return the length of the longest palindrome that can be built with those letters. Letters are case sensitive, for example, "Aa" is not considered a palindrome here.

**Constraints:** 
1 <= s.length <= 2000
s consists of lowercase and/or uppercase English letters only.

**Examples:** 
```python3
s = "abccccdd" #=> 7
s = "a" #=> 1
```

**Hint:** Use a hashMap of character frequencies. Iterate over values. If value is even, add it to total. If value is odd, add it - 1 to the total. At the end return total if no odd found and total + 1 if odd found

```python3
class Solution(object):
    def longestPalindrome(self, s):
        hash = set()
        for c in s:
            if c not in hash:
                hash.add(c)
            else:
                hash.remove(c)
        # len(hash) is the number of the odd letters
        return len(s) - len(hash) + 1 if len(hash) > 0 else len(s)
```
**Time:** O(n)
**Space:** O(1)

## 56. Longest Common Prefix
**Reference:** https://leetcode.com/problems/longest-common-prefix/solutions/3273176/python3-c-java-19-ms-beats-99-91/

**Description:** Write a function to find the longest common prefix string amongst an array of strings. If there is no common prefix, return an empty string "".

**Constraints:** 
1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters.

**Examples:** 
```python3
strs = ["flower","flow","flight"] #=> "fl"
strs = ["dog","racecar","car"] #=> ""
```

**Hint:** Sort the input list v lexicographically. "If the array is sorted alphabetically then you can assume that the first element of the array and the last element of the array will have most different prefixes of all comparisons that could be made between strings in the array. If this is true, you only have to compare these two strings."

```python3
class Solution:
    def longestCommonPrefix(self, v: List[str]) -> str:
        ans=""
        v=sorted(v)
        first=v[0]
        last=v[-1]
        for i in range(min(len(first),len(last))):
            if(first[i]!=last[i]):
                return ans
            ans+=first[i]
        return ans 
```
**Time:** O(nlog(n))
**Space:** O(1)

## 57. Longest Substring Without Repeating Characters
**Reference:** https://leetcode.com/problems/longest-substring-without-repeating-characters/solutions/127839/longest-substring-without-repeating-characters/

**Description:** Given a string, find the length of the longest substring without repeating characters.

**Constraints:** 
1 <= strs.length <= 200
0 <= strs[i].length <= 200
strs[i] consists of only lowercase English letters.

**Examples:** 
```python3
"abcabcbb" #=> 3
"bbbbb" #=> 1
"pwwkew" #=> 3
```

**Hint:** Note: substrings don't get reordered. Use sliding window and a hashMap. Start with left ptrs at index 0. Map the curr char to current index of i. If hashMap already has current char, increment i to one index after hashMap[currentChar]. Update global max if possible globalMax = max(globalMax, j - i + 1). Increment hashMap[currentChar] to j + 1

```javascript
const lengthOfLongestSubstring = (s) => {
  let max = 0;
  const hashMap = {};
  for (let j = 0, i = 0; j < s.length; j++) {
    const char = s[j];
    if (hashMap[char]) i = Math.max(hashMap[char], i) // skip chars
    max = Math.max(max, j - i + 1);
    hashMap[char] = j + 1; // Map character to its index
  }
  return max
};
```
**Time:** O(n)
**Space:** O(min(m, n)) n = string size, n = charset size

## 58. String to Integer (atoi)
**Reference:** https://leetcode.com/problems/string-to-integer-atoi/solutions/425289/python-99-89-no-cheating-by-using-int/

**Description:** Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).

The algorithm for myAtoi(string s) is as follows:

1. Read in and ignore any leading whitespace.
2. Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
3. Read in next the characters until the next non-digit character or the end of the input is reached. The rest of the string is ignored.
4. Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
5. If the integer is out of the 32-bit signed integer range [-2^31, 2^31 - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -2^31 should be clamped to -2^31, and integers greater than 2^31 - 1 should be clamped to 2^31 - 1.
6. Return the integer as the final result.

Note:
1. Only the space character ' ' is considered a whitespace character.
2. Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.

**Constraints:** 
0 <= s.length <= 200
s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.

**Examples:** 
```python3
s = "42" #=> 42
s = "   -42" #=> -42
s = "4193 with words" #=> 4193
```

**Hint:** Remove leading whitespace. Note the sign. Use hashMap to conver digits to numbers. Check for overflow.

```python3
MAPPING = {
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "0": 0,
}

MAX_INT = 2**31-1
MIN_INT = -(2**31)

class Solution:
    def myAtoi(self, string: str) -> int:
        s = string.lstrip(' ')
        if not s:
            return 0
        
        sign = -1 if s[0] == "-" else 1
        if sign != 1 or s[0] == "+":
            s = s[1:]
            
        res = 0
        for c in s:
            if c not in MAPPING:
                return self.limit(res * sign)
            
            res *= 10
            res += MAPPING[c]
            
        return self.limit(res * sign)
    
    def limit(self, x: int) -> int:
        if x > MAX_INT:
            return MAX_INT
        if x < MIN_INT:
            return MIN_INT
        return x
```
**Time:** O(n)
**Space:** O(1)

## 59. Longest Palindromic Substring
**Reference:** https://leetcode.com/problems/longest-palindromic-substring/solutions/2928/Very-simple-clean-java-solution/

**Description:** Given a string s, return the longest palindromic substring in s.

**Constraints:** 
1 <= s.length <= 1000
s consist of only digits and English letters.

**Examples:** 
```python3
s = "babad" #=> "bab"
s = "cbbd" #=> "bb"
```

**Hint:** For each element in array, try to grow left and right from current index (odd) or curr index and curr index + 1 (even)

```python3
class Solution:
    def longestPalindrome(self, s: str) -> str:
        self.lo, self.maxLen = 0, 0

        if len(s) < 2:
            return s

        for i in range(len(s) - 1):
            self.extendPalindrome(s, i, i)  # assume odd length, try to extend Palindrome as possible
            self.extendPalindrome(s, i, i + 1)  # assume even length

        return s[self.lo:self.lo + self.maxLen]

    def extendPalindrome(self, s: str, j: int, k: int) -> None:
        while j >= 0 and k < len(s) and s[j] == s[k]:
            j -= 1
            k += 1
        if self.maxLen < k - j - 1:
            self.lo = j + 1
            self.maxLen = k - j - 1
```
**Time:** O(n^2)
**Space:** O(n) ??

## 60. Find All Anagrams in a String
**Reference:** https://leetcode.com/problems/find-all-anagrams-in-a-string/solutions/1737985/python3-sliding-window-hash-table-explained/

**Description:** Given two strings s and p, return an array of all the start indices of p's anagrams in s. You may return the answer in any order. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Constraints:** 
1 <= s.length, p.length <= 3 * 10^4
s and p consist of lowercase English letters.

**Examples:** 
```python3
s = "cbaebabacd", p = "abc" #=> [0,6]
s = "abab", p = "ab" #=> [0,1,2]
```

**Hint:** "First, we have to create a hash map with letters from p as keys and its frequencies as values. Then, we start sliding the window [0..len(s)] over the s. Every time a letter gets out of the window, we increase the corresponding counter in the hashmap, and when a letter gets in the window - we decrease. As soon as all counters in the hashmap become zeros we encountered an anagram."

```python3
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        hm, res, pL, sL = defaultdict(int), [], len(p), len(s)
        if pL > sL: return []

        # build hashmap
        for ch in p: hm[ch] += 1

        # initial full pass over the window
        for i in range(pL-1):
            if s[i] in hm: hm[s[i]] -= 1
            
        # slide the window with stride 1
        for i in range(-1, sL-pL+1):
            if i > -1 and s[i] in hm:
                hm[s[i]] += 1
            if i+pL < sL and s[i+pL] in hm: 
                hm[s[i+pL]] -= 1
                
            # check whether we encountered an anagram
            if all(v == 0 for v in hm.values()): 
                res.append(i+1)
                
        return res        
```
**Time:** O(n)
**Space:** O(1)

## 61. Group Anagrams
**Reference:** https://leetcode.com/problems/group-anagrams/solutions/2384037/python-easily-understood-hash-table-fast-simple/

**Description:** Given an array of strings strs, group the anagrams together. You can return the answer in any order. An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

**Constraints:** 
1 <= strs.length <= 10^4
0 <= strs[i].length <= 100
strs[i] consists of lowercase English letters.

**Examples:** 
```python3
strs = ["eat","tea","tan","ate","nat","bat"] #=> [["bat"],["nat","tan"],["ate","eat","tea"]]
strs = [""] #=> [[""]]
strs = ["a"] #=> [["a"]]
```

**Hint:** Use hash table. The key is the character count the values are the strings themselves. Return the hash table values

```python3
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        strs_table = {}

        for string in strs:
            sorted_string = ''.join(sorted(string))

            if sorted_string not in strs_table:
                strs_table[sorted_string] = []

            strs_table[sorted_string].append(string)

        return list(strs_table.values())   
```
**Time:** O(m*nlogn))
**Space:** O(n)

## 62. Longest Repeating Character Replacement
**Reference:** https://leetcode.com/problems/longest-repeating-character-replacement/solutions/91271/java-12-lines-o-n-sliding-window-solution-with-explanation/

**Description:** You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times. Return the length of the longest substring containing the same letter you can get after performing the above operations.

**Constraints:** 
1 <= s.length <= 10^5
s consists of only uppercase English letters.
0 <= k <= s.length

**Examples:** 
```python3
s = "ABAB", k = 2 #=> 4
s = "AABABBA", k = 1 #=> 4
```

**Hint:** Use sliding window starting with L and R ptrs at idx 0. Use a map to count the number of current char. If the length of the window - the freq of the most frequent char in window is <= k, keep moving right ptr right. Else move left ptr until valid again. Return largest window

```python3
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        count = [0] * 26
        start = 0
        maxCount = 0
        maxLength = 0

        for end in range(len(s)):
            count[ord(s[end]) - ord('A')] += 1
            maxCount = max(maxCount, count[ord(s[end]) - ord('A')])

            if end - start + 1 - maxCount > k:
                count[ord(s[start]) - ord('A')] -= 1
                start += 1

            maxLength = max(maxLength, end - start + 1)

        return maxLength  
```
**Time:** O(((N + 26) * N) * (M - N))
**Space:** O(1)

## 63. Largest Number
**Reference:** https://leetcode.com/problems/largest-number/solutions/1012321/javascript-with-sort-o-nlogn/

**Description:** Given a list of non-negative integers nums, arrange them such that they form the largest number and return it. Since the result may be very large, so you need to return a string instead of an integer.

**Constraints:** 
1 <= nums.length <= 100
0 <= nums[i] <= 10^9

**Examples:** 
```python3
nums = [10,2] #=> "210"
nums = [3,30,34,5,9] #=> "9534330"
```

**Hint:** Use custom sort that compares the concatenated order of each character. Then join

```python3
class Solution:
    def largestNumber(self, nums: List[int]) -> str:
        if not nums or len(nums) == 0:
            return '0'
        nums = sorted(map(str, nums), key=lambda x: x * 9, reverse=True)
        if nums[0] == '0':
            return '0'
        return ''.join(nums)
```
**Time:** O(nlogn)
**Space:** O(n)

## 64. Encode and Decode Strings
**Reference:** https://leetcode.com/problems/largest-number/solutions/1012321/javascript-with-sort-o-nlogn/

**Description:** Design an algorithm to encode a list of strings to a string. The encoded string is then sent over the network and is decoded back to the original list of strings.

Machine 1 (sender) has the function:
```code
string encode(vector<string> strs) {
  // ... your code
  return encoded_string;
}
```

Machine 2 (receiver) has the function:
```code
vector<string> decode(string s) {
  //... your code
  return strs;
}
``

So Machine 1 does:
```code
string encoded_string = encode(strs);
```

and Machine 2 does:
```code 
vector<string> strs2 = decode(encoded_string);
```

strs2 in Machine 2 should be the same as strs in Machine 1.

Implement the encode and decode methods.

Note:
1. The string may contain any possible characters out of 256 valid ascii characters. Your algorithm should be generalized enough to work on any possible characters.
2. Do not use class member/global/static variables to store states. Your encode and decode algorithms should be stateless.
3. Do not rely on any library method such as eval or serialize methods. You should implement your own encode/decode algorithm.

**Constraints:** 
1 <= nums.length <= 100
0 <= nums[i] <= 10^9

**Examples:** 
```python3
nums = [10,2] #=> "210"
nums = [3,30,34,5,9] #=> "9534330"
```

**Hint:** Create a delimiter for your encode method. She uses / and the length of the word to avoid collisions with real characters.

```python3
class Solution:
    def encode(self, strs):
        res = ''
        for s in strs:
            encoded = str(len(s)) + '/' + s
            res += encoded
        return res
    def decode(self, str):
        res, i = [], 0
        while i < len(str):
            # For example, 12/abc
            e = i
            while e < len(str) and str[e] != '/':
                e += 1
            size = int(str[i:e])
            word = str[e + 1, e + 1 + size]
            i = e + 1 + size
            res.append(word)
        return res
```
**Time:** O(n)
**Space:** O(n)

## 65. Minimum Window Substring
**Reference:** https://leetcode.com/problems/minimum-window-substring/solutions/26808/here-is-a-10-line-template-that-can-solve-most-substring-problems/

**Description:** Given two strings s and t of lengths m and n respectively, return the minimum window substring of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "". The testcases will be generated such that the answer is unique. Follow up: Could you find an algorithm that runs in O(m + n) time?

**Constraints:** 
m == s.length
n == t.length
1 <= m, n <= 10^5
s and t consist of uppercase and lowercase English letters.

**Examples:** 
```python3
s = "ADOBECODEBANC", t = "ABC" #=> "BANC"
s = "a", t = "a" #=> "a"
s = "a", t = "aa" #=> ""
```

**Hint:** use a hashmap assisted with two pointers
"1. Use two pointers: start and end to represent a window.
2. Move end to find a valid window.
3. When a valid window is found, move start to find a smaller window."

```java
class Solution {
  public String minWindow(String s, String t) {
    int [] map = new int[128];
    for (char c : t.toCharArray()) {
      map[c]++;
    }
    int start = 0, end = 0, minStart = 0, minLen = Integer.MAX_VALUE, counter = t.length();
    while (end < s.length()) {
      final char c1 = s.charAt(end);
      if (map[c1] > 0) counter--;
      map[c1]--;
      end++;
      while (counter == 0) {
        if (minLen > end - start) {
          minLen = end - start;
          minStart = start;
        }
        final char c2 = s.charAt(start);
        map[c2]++;
        if (map[c2] > 0) counter++;
        start++;
      }
    }

    return minLen == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLen);
  }
}
```
**Time:** ???
**Space:** ???

## 66. Palindrome Pairs
**Reference:** https://leetcode.com/problems/palindrome-pairs/solutions/79210/The-Easy-to-unserstand-JAVA-Solution/

**Description:** You are given a 0-indexed array of unique strings words.
A palindrome pair is a pair of integers (i, j) such that:
1. 0 <= i, j < words.length,
2. i != j, and
3. words[i] + words[j] (the concatenation of the two strings) is a palindrome.
Return an array of all the palindrome pairs of words.
You must write an algorithm with O(sum of words[i].length) runtime complexity.

**Constraints:** 
1 <= words.length <= 5000
0 <= words[i].length <= 300
words[i] consists of lowercase English letters.

**Examples:** 
```python3
words = ["abcd","dcba","lls","s","sssll"] #=> [[0,1],[1,0],[3,2],[2,4]]
words = ["bat","tab","cat"] #=> [[0,1],[1,0]]
words = ["a",""] #=> [[0,1],[1,0]]
```

**Hint:** 
Case1: If s1 is a blank string, then for any string that is palindrome s2, s1+s2 and s2+s1 are palindrome.
Case 2: If s2 is the reversing string of s1, then s1+s2 and s2+s1 are palindrome.
Case 3: If s1[0:cut] is palindrome and there exists s2 is the reversing string of s1[cut+1:] , then s2+s1 is palindrome.
Case 4: Similiar to case3. If s1[cut+1: ] is palindrome and there exists s2 is the reversing string of s1[0:cut] , then s1+s2 is palindrome.
To make the search faster, build a HashMap to store the String-idx pairs.

```java
public class Solution {
public List<List<Integer>> palindromePairs(String[] words) {
    List<List<Integer>> res = new ArrayList<List<Integer>>();
    if(words == null || words.length == 0){
        return res;
    }
    //build the map save the key-val pairs: String - idx
    HashMap<String, Integer> map = new HashMap<>();
    for(int i = 0; i < words.length; i++){
        map.put(words[i], i);
    }
    
    //special cases: "" can be combine with any palindrome string
    if(map.containsKey("")){
        int blankIdx = map.get("");
        for(int i = 0; i < words.length; i++){
            if(isPalindrome(words[i])){
                if(i == blankIdx) continue;
                res.add(Arrays.asList(blankIdx, i));
                res.add(Arrays.asList(i, blankIdx));
            }
        }
    }
    
    //find all string and reverse string pairs
    for(int i = 0; i < words.length; i++){
        String cur_r = reverseStr(words[i]);
        if(map.containsKey(cur_r)){
            int found = map.get(cur_r);
            if(found == i) continue;
            res.add(Arrays.asList(i, found));
        }
    }
    
    //find the pair s1, s2 that 
    //case1 : s1[0:cut] is palindrome and s1[cut+1:] = reverse(s2) => (s2, s1)
    //case2 : s1[cut+1:] is palindrome and s1[0:cut] = reverse(s2) => (s1, s2)
    for(int i = 0; i < words.length; i++){
        String cur = words[i];
        for(int cut = 1; cut < cur.length(); cut++){
            if(isPalindrome(cur.substring(0, cut))){
                String cut_r = reverseStr(cur.substring(cut));
                if(map.containsKey(cut_r)){
                    int found = map.get(cut_r);
                    if(found == i) continue;
                    res.add(Arrays.asList(found, i));
                }
            }
            if(isPalindrome(cur.substring(cut))){
                String cut_r = reverseStr(cur.substring(0, cut));
                if(map.containsKey(cut_r)){
                    int found = map.get(cut_r);
                    if(found == i) continue;
                    res.add(Arrays.asList(i, found));
                }
            }
        }
    }
    
    return res;
}

public String reverseStr(String str){
    StringBuilder sb= new StringBuilder(str);
    return sb.reverse().toString();
}

public boolean isPalindrome(String s){
    int i = 0;
    int j = s.length() - 1;
    while(i <= j){
        if(s.charAt(i) != s.charAt(j)){
            return false;
        }
        i++;
        j--;
    }
    return true;
}
}
```
**Time:** ???
**Space:** ???

## 67. Invert Binary Tree
**Reference:** https://leetcode.com/problems/invert-binary-tree/solutions/62714/3-4-lines-python/

**Description:** Given the root of a binary tree, invert the tree, and return its root.

**Constraints:** 
The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100

**Examples:** 
```python3
root = [4,2,7,1,3,6,9] #=> [4,7,2,9,6,3,1]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/e2ec979c-6302-48a1-8262-35c1f2a24f77)


```python3
root = [2,1,3] #=> [2,3,1]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/4851be24-ae71-4ac0-91e3-6e9c9c9ea8fa)


```python3
root = [] #=> []
```

**Hint:** Similar to normal recursive DFS except, call recurse on left and right and save the result then swap the results and return root at end

```python3
def invertTree(self, root):
    if root:
        invert = self.invertTree
        root.left, root.right = invert(root.right), invert(root.left)
        return root
```
**Time:** O(n)
**Space:** O(n)

## 68. Balanced Binary Tree
**Reference:** https://leetcode.com/problems/balanced-binary-tree/solutions/35691/the-bottom-up-o-n-solution-would-be-better/

**Description:** Given a binary tree, determine if it is height-balanced.

**Constraints:** 
The number of nodes in the tree is in the range [0, 5000].
-10^4 <= Node.val <= 10^4

**Examples:** 
```python3
root = [1,2,2,3,3,null,null,4,4] #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/1b563df0-94b1-4b1f-bd6c-6a7c8db17461)


```python3
root = [3,9,20,null,null,15,7] #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/c1645ec2-e88d-4d42-9e93-f87b717c3185)


```python3
root = [] #=> true
```

**Hint:** Balanced if no two subtrees differ in height by more than 1.  Use bottom up DFS, bubbling up -1 if the height don't match or just check the height at each node (O(n^2))

```cpp
class solution {
public:
int dfsHeight (TreeNode *root) {
        if (root == NULL) return 0;
        
        int leftHeight = dfsHeight (root -> left);
        if (leftHeight == -1) return -1;
        int rightHeight = dfsHeight (root -> right);
        if (rightHeight == -1) return -1;
        
        if (abs(leftHeight - rightHeight) > 1)  return -1;
        return max (leftHeight, rightHeight) + 1;
    }
    bool isBalanced(TreeNode *root) {
        return dfsHeight (root) != -1;
    }
};
```
**Time:** O(n)
**Space:** O(1)

## 69. Diameter of Binary Tree
**Reference:** https://leetcode.com/problems/diameter-of-binary-tree/solutions/1102557/diameter-of-binary-tree/

**Description:** Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root. Note: The length of path between two nodes is represented by the number of edges between them.

**Constraints:** 
???

**Examples:** 
```python3
          1
         / \
        2   3
       / \
      4   5
#=> 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
```

**Hint:** Return the longest path between any two nodes. You don't have to pass throught the root. The solution is essentially the same as the height algorithm, except you are updating a global diameter variable each iteration rightHeight + leftHeight is greater than it. Tim note, would this actaully be n^2??

```python3
class Solution:
    def diameterOfBinaryTree(self, root):
        self.diameter = 0

        def howHigh(node):
            if not node:
                return 0
            left = howHigh(node.left)
            right = howHigh(node.right)
            self.diameter = max(self.diameter, left + right)
            return max(left, right) + 1

        howHigh(root)
        return self.diameter
```
**Time:** O(n)
**Space:** O(n)

## 70. Maximum Depth of Binary Tree
**Reference:** https://leetcode.com/problems/maximum-depth-of-binary-tree/solutions/1770060/c-recursive-dfs-example-dry-run-well-explained/

**Description:** Given the root of a binary tree, return its maximum depth. A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

**Constraints:** 
The number of nodes in the tree is in the range [0, 10^4].
-100 <= Node.val <= 100

**Examples:** 
```python3
root = [3,9,20,null,null,15,7] #=> 3
```

![image](https://github.com/will4skill/algo-review/assets/10373005/392bc241-239d-4959-8c4a-73ad06c1eb21)

```python3
root = [1,null,2] #=> 2
```

**Hint:** Similar to normal recursive DFS except, call recurse on left and right and save the result then swap the results and return root at end

```python3
class Solution(object):
    def maxDepth(self, root):
        if not root: return 0
        maxLeft = self.maxDepth(root.left)
        maxRight = self.maxDepth(root.right)
        return max(maxLeft, maxRight) + 1
```
**Time:** O(n)
**Space:** O(height of tree)

## 71. Same Tree
**Reference:** https://leetcode.com/problems/same-tree/solutions/642761/easy-to-understand-faster-simple-recursive-iterative-dfs-python-solution/

**Description:** Given the roots of two binary trees p and q, write a function to check if they are the same or not. Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.

**Constraints:** 
The number of nodes in both trees is in the range [0, 100].
-10^4 <= Node.val <= 10^4

**Examples:** 
```python3
p = [1,2,3], q = [1,2,3] #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/71982055-57d7-4145-9086-fa0344ed1f51)

```python3
p = [1,2], q = [1,null,2] #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/6ab69227-f462-40a0-be01-854409c52e59)

```python3
p = [1,2,1], q = [1,1,2] #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/e169d27c-ece5-4034-9580-587f5af77a03)


**Hint:** Recursive or iterative. If curr.left and curr.right == null that's fine. If One is null, or the values of left and right are not equal return false. Iterate over tree.

```python3
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q:
            return False
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```
**Time:** O(n)
**Space:** O(height of tree)

## 72. Symmetric Tree
**Reference:** https://leetcode.com/problems/symmetric-tree/solutions/33050/recursively-and-iteratively-solution-in-python/

**Description:** Given the root of a binary tree, check whether it is a mirror of itself (i.e., symmetric around its center). Follow up: Could you solve it both recursively and iteratively?

**Constraints:** 
The number of nodes in the tree is in the range [1, 1000].
-100 <= Node.val <= 100

**Examples:** 
```python3
root = [1,2,2,3,4,4,3] #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/2ed6370d-093c-471a-a02b-c8614d17fffc)


```python3
root = [1,2,2,null,3,null,3] #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/5a5e86c4-3059-415c-8db8-ed03bc28edfa)


**Hint:** If you take the recursive approach, you need a helper function to compare nodes from left and right branches simultaneously. Remember to compare left.left with right.right and left.right with right.left.

```python3
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if root is None:
            return True
        return self.isMirror(root.left, root.right)

    def isMirror(self, left, right):
        if left is None and right is None:
            return True
        if left is None or right is None:
            return False
        if left.val != right.val:
            return False

        outPair = self.isMirror(left.left, right.right)
        inPair = self.isMirror(left.right, right.left)
        return outPair and inPair
```
**Time:** O(n)
**Space:** O(height of tree)

## 73. Subtree of Another Tree
**Reference:** https://leetcode.com/problems/subtree-of-another-tree/

**Description:** Given the roots of two binary trees root and subRoot, return true if there is a subtree of root with the same structure and node values of subRoot and false otherwise. A subtree of a binary tree tree is a tree that consists of a node in tree and all of this node's descendants. The tree tree could also be considered as a subtree of itself.

**Constraints:** 
The number of nodes in the root tree is in the range [1, 2000].
The number of nodes in the subRoot tree is in the range [1, 1000].
-10^4 <= root.val <= 10^4
-10^4 <= subRoot.val <= 10^4

**Examples:** 
```python3
root = [3,4,5,1,2], subRoot = [4,1,2] #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/8a41c663-af13-4fa8-ab78-1e1274debff0)


```python3
root = [3,4,5,1,2,null,null,null,null,0], subRoot = [4,1,2] #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/9170b092-9dac-4585-88fe-f42ce84402e5)


**Hint:** Recursive. Iterate overtree, calling the isSame method on each node. If same subtree is found, return true. The final statement compares the left and right branches if one or the other has a matching subtree, return true.

```python3
class Solution:
    def isEqual(self, rootA, rootB):
        if rootA == None and rootB != None or rootA != None and rootB == None:
            return False
        if rootA == None and rootB == None:
            return True

        return rootA.val == rootB.val and self.isEqual(rootA.left, rootB.left) and self.isEqual(rootA.right, rootB.right)

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        if root == None:
            return False
        if self.isEqual(root, subRoot): return True

        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

```
**Time:** O(S*T)
**Space:** O(height of taller tree)

## 74. Binary Tree Level Order Traversal
**Reference:** https://leetcode.com/problems/binary-tree-level-order-traversal/description/

**Description:** Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

**Constraints:** 
The number of nodes in the tree is in the range [0, 2000].
-1000 <= Node.val <= 1000

**Examples:** 
```python3
root = [3,9,20,null,null,15,7] #=> [[3],[9,20],[15,7]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d1585565-a7ef-4344-a872-8e4183e8cbc5)

```python3
root = [1] #=> [[1]]
```

```python3
root = [] #=> []
```

**Hint:** BFS: use a queue that tracks node and level. Maintain a level array. If curre level is in level array add node, else add new level to level array then add node. Traverse while increasing level number
DFS: Similar logic to above with the level array, remember to increment levelNumber each time you recurse

```python3
# https://leetcode.com/problems/binary-tree-level-order-traversal/solutions/33550/python-solution-with-detailed-explanation/
class Solution(object):
    def levelOrder(self, root):
        result = []
        self.helper(root, 0, result)
        return result
    
    def helper(self, root, level, result):
        if root is None:
            return
        if len(result) <= level:
            result.append([])
        result[level].append(root.val)
        self.helper(root.left, level+1, result)
        self.helper(root.right, level+1, result)
```

```python3
# https://leetcode.com/problems/binary-tree-level-order-traversal/solutions/1219538/python-simple-bfs-explained/
class Solution:
    def levelOrder(self, root):
        if not root: return []
        queue, result = deque([root]), []
        
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level)
        return result
```
**Time:** O(n)
**Space:** O(n)

## 75. Lowest Common Ancestor of a Binary Tree
**Reference:** https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/solutions/152682/python-simple-recursive-solution-with-detailed-explanation/

**Description:** Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree. According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

**Constraints:** 
The number of nodes in the tree is in the range [2, 10^5].
-10^9 <= Node.val <= 10^9
All Node.val are unique.
p != q
p and q will exist in the tree.

**Examples:** 

```python3
root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1 #=> 3
```

![image](https://github.com/will4skill/algo-review/assets/10373005/f012ca04-eb47-41d3-b51a-16a9758c5037)

```python3
root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4 #=> 5
```

![image](https://github.com/will4skill/algo-review/assets/10373005/e24c2c4a-055d-4b1e-bc7f-87c24a9bda55)


```python3
root = [1,2], p = 1, q = 2 #=> 1
```

**Hint:** Recursively, find a path to target 1, find a path to target 2. Put those path arrays into a set. Traverse the other array leaf up and return the first intersecting node. 
See below for an alternative approach:

```python3
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # If looking for me, return myself
        if root == p or root == q:
            return root
        left = right = None
        # else look in left and right child
        if root.left:
            left = self.lowestCommonAncestor(root.left, p, q)
        if root.right:
            right = self.lowestCommonAncestor(root.right, p, q)
        # if both children returned a node, means
        # both p and q found so parent is LCA
        if left and right:
            return root
        # either one of the chidren returned a node, meaning either p or q found on left or right branch.
        # Example: assuming 'p' found in left child, right child returned 'None'. This means 'q' is
        # somewhere below node where 'p' was found we dont need to search all the way, 
        # because in such scenarios, node where 'p' found is LCA
        return left or right
```
**Time:** O(n)
**Space:** O(h)

## 76. Binary Tree Right Side View
**Reference:** https://leetcode.com/problems/binary-tree-right-side-view/solutions/56012/my-simple-accepted-solution-java/

**Description:** Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

**Constraints:** 
The number of nodes in the tree is in the range [0, 100].
-100 <= Node.val <= 100

**Examples:** 

```python3
root = [1,2,3,null,5,null,4] #=> [1,3,4]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/477bf0e3-4eda-4288-8fa5-b95176e2135e)


```python3
root = [1,null,3] #=> [1,3]
```

```python3
root = [] #=> []
```

**Hint:** This is essentially level order traversal, but you only add to the output array if your current depth equals the size of the result array.
*the key is to do rightView(curr.right, result, currDepth + 1); first...

```python3
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        self.rightView(root, result, 0)
        return result

    def rightView(self, curr, result, currDepth):
        if curr is None:
            return
        if currDepth == len(result):
            result.append(curr.val)
        self.rightView(curr.right, result, currDepth + 1)
        self.rightView(curr.left, result, currDepth + 1)
```
**Time:** O(n)
**Space:** O(h)

## 77. Construct Binary Tree from Preorder and Inorder Traversal
**Reference:** https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/solutions/565412/detailed-python-walkthrough-from-an-o-n-2-solution-to-o-n-faster-than-99-77/

**Description:** Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

**Constraints:** 
1 <= preorder.length <= 3000
inorder.length == preorder.length
-3000 <= preorder[i], inorder[i] <= 3000
preorder and inorder consist of unique values.
Each value of inorder also appears in preorder.
preorder is guaranteed to be the preorder traversal of the tree.
inorder is guaranteed to be the inorder traversal of the tree.

**Examples:** 

```python3
preorder = [3,9,20,15,7], inorder = [9,3,15,20,7] #=> [3,9,20,null,null,15,7]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/bd2a041b-e622-4885-8656-e597059fe5ae)

```python3
preorder = [-1], inorder = [-1] #=> [-1]
```

**Hint:** To do this in place, maintain a start and end idx for each array. Create a new root with preorder[0]. Find the mid Idx with new root val and inorder array. Just as before, create new left and right children for the new node by splitting the inorder and preorder arrays (idx wise). Return root.

```python3
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        preorder.reverse() # You can also convert to a deque. The point is for O(1) pop operations
        inorderDict = { v:i for i,v in enumerate(inorder) } # This provides O(1) idx lookups
        return self.buildTreeHelper(preorder, inorderDict, 0, len(preorder) - 1)

    def buildTreeHelper(self, preorder, inorderDict, beg, end):
        if beg > end: return None
        root = TreeNode(preorder.pop())
        index = inorderDict[root.val]
        
        root.left = self.buildTreeHelper(preorder, inorderDict, beg, index - 1)
        root.right = self.buildTreeHelper(preorder, inorderDict, index + 1, end)
        return root
```
**Time:** O(n)
**Space:** O(n)

## 78. Path Sum II
**Reference:** https://leetcode.com/problems/path-sum-ii/solutions/2615948/leetcode-the-hard-way-explained-line-by-line/

**Description:** Given the root of a binary tree and an integer targetSum, return all root-to-leaf paths where the sum of the node values in the path equals targetSum. Each path should be returned as a list of the node values, not node references.

A root-to-leaf path is a path starting from the root and ending at any leaf node. A leaf is a node with no children.

**Constraints:** 
The number of nodes in the tree is in the range [0, 5000].
-1000 <= Node.val <= 1000
-1000 <= targetSum <= 1000

**Examples:** 

```python3
root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22 #=> [[5,4,11,2],[5,8,4,5]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/033f984a-0ce4-4cbd-b500-295a6ee8aa4d)

```python3
root = [1,2,3], targetSum = 5 #=> []
```

![image](https://github.com/will4skill/algo-review/assets/10373005/8469565a-3a3d-4c7a-851d-fe6f7ed050aa)

```python3
root = [1,2], targetSum = 0 #=> []
```

**Hint:** DFS and backtracking to avoid array copies. At leaf, check if sum == node.val. While you recurse, subract node.val from sum

```python3
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        ans = []
        self.dfs(root, [], ans, targetSum)
        return ans
    def dfs(self, root, path, ans, remainingSum):
        if not root:
            return
        path.append(root.val)
        if not root.left and not root.right and remainingSum == root.val:
            ans.append(list(path))
        self.dfs(root.left, path, ans, remainingSum - root.val)
        self.dfs(root.right, path, ans, remainingSum - root.val)
        path.pop() # backtrack 
```
**Time:** O(n)
**Space:** O(height of tree)

## 79. Maximum Width of Binary Tree
**Reference:** https://leetcode.com/problems/maximum-width-of-binary-tree/solutions/3436593/image-explanation-why-long-to-int-c-java-python/

**Description:** Given the root of a binary tree, return the maximum width of the given tree. The maximum width of a tree is the maximum width among all levels. The width of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes), where the null nodes between the end-nodes that would be present in a complete binary tree extending down to that level are also counted into the length calculation. It is guaranteed that the answer will in the range of a 32-bit signed integer.

**Constraints:** 
The number of nodes in the tree is in the range [1, 3000].
-100 <= Node.val <= 100

**Examples:** 

```python3
root = [1,3,2,5,3,null,9] #=> 4
```

![image](https://github.com/will4skill/algo-review/assets/10373005/e1d8179e-9b7e-4ba4-b450-2dbf6cc8ef58)

```python3
root = [1,3,2,5,null,null,9,6,null,7] #=> 7
```

![image](https://github.com/will4skill/algo-review/assets/10373005/4b8c0c23-8c0e-4eea-be39-72bd1c156d62)

```python3
root = [1,3,2,5] #=> 2
```

![image](https://github.com/will4skill/algo-review/assets/10373005/70aa7c95-8bd1-4c83-9712-fc5a9f46eb02)


**Hint:** Combine level order BFS with a queue that stores the node and placement in a tuple.

```python3
class Solution:
    def widthOfBinaryTree(self, root: TreeNode) -> int:
        if not root:
            return 0
        queue = deque([(root, 0)])
        max_width = 0
        
        while queue:
            level_length = len(queue)
            level_start = queue[0][1]
            
            for i in range(level_length):
                node, index = queue.popleft() # Destructure
                
                if node.left:
                    queue.append((node.left, 2*index))
                if node.right:
                    queue.append((node.right, 2*index+1))
                    
            max_width = max(max_width, index - level_start + 1)
            
        return max_width
```
**Time:** O(n)
**Space:** O(n)

## 80. Binary Tree Zigzag Level Order Traversal
**Reference:** https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/solutions/749036/python-clean-bfs-solution-explained/

**Description:** Given the root of a binary tree, return the zigzag level order traversal of its nodes' values. (i.e., from left to right, then right to left for the next level and alternate between).

**Constraints:** 
The number of nodes in the tree is in the range [0, 2000].
-100 <= Node.val <= 100

**Examples:** 

```python3
root = [3,9,20,null,null,15,7] #=> [[3],[20,9],[15,7]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d9386e51-b9a6-4547-a4c1-1038cadbb409)

```python3
root = [1] #=> [[1]]
```

```python3
root = [] #=> []
```

**Hint:** Traverse in level order, but track level number (even or odd) and alternate order

```python3
class Solution:
    def zigzagLevelOrder(self, root):
        if not root: return []
        queue = deque([root])
        result, direction = [], 1
        
        while queue:
            level = []
            for i in range(len(queue)):
                node = queue.popleft()
                level.append(node.val)
                if node.left:  queue.append(node.left)
                if node.right: queue.append(node.right)
            result.append(level[::direction])
            direction *= (-1)
        return result
```
**Time:** O(n)
**Space:** O(n)

## 81. Path Sum III
**Reference:** https://leetcode.com/problems/path-sum-iii/solutions/141424/python-step-by-step-walk-through-easy-to-understand-two-solutions-comparison/

**Description:** Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values along the path equals targetSum. The path does not need to start or end at the root or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).

**Constraints:** 
The number of nodes in the tree is in the range [0, 1000].
-10^9 <= Node.val <= 10^9
-1000 <= targetSum <= 1000

**Examples:** 

```python3
root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8 #=> 3
```

![image](https://github.com/will4skill/algo-review/assets/10373005/50b0d663-6bbd-4455-a6e8-e3fd480bef8a)


```python3
root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22 #=> 3
```

**Hint:** In order to optimize from the brutal force solution, we will have to think of a clear way to memorize the intermediate result. Namely in the brutal force solution, we did a lot repeated calculation. For example 1->3->5, we calculated: 1, 1+3, 1+3+5, 3, 3+5, 5.
This is a classical 'space and time tradeoff': we can create a dictionary (named cache) which saves all the path sum (from root to current node) and their frequency.
Again, we traverse through the tree, at each node, we can get the currPathSum (from root to current node). If within this path, there is a valid solution, then there must be a oldPathSum such that currPathSum - oldPathSum = target.
We just need to add the frequency of the oldPathSum to the result.
During the DFS break down, we need to -1 in cache[currPathSum], because this path is not available in later traverse.

```python3
    def pathSum(self, root, target):
        # define global result and path
        self.result = 0
        
        self.cache ={}
        
        # recursive to get result
        self.dfs(root, target, 0)
        
        # return result
        return self.result
    
    def dfs(self, root, target, curr_sum):
        if not root:
            return None
        
        curr_sum = curr_sum+root.val
        
        if curr_sum == target :
            self.result+=1
            
        if (curr_sum-target) in self.cache:
            self.result += self.cache[curr_sum-target]
            
        if curr_sum in self.cache:
            self.cache[curr_sum] +=1
        else:
            self.cache[curr_sum] = 1
        
        self.dfs(root.left, target, curr_sum)
        self.dfs(root.right, target, curr_sum)
        # when move to a different branch, the currPathSum is no longer available, hence remove one.
        self.cache[curr_sum] -=1
```
**Time:** O(n)
**Space:** O(n)

## 82. All Nodes Distance K in Binary Tree
**Reference:** https://leetcode.com/problems/all-nodes-distance-k-in-binary-tree/solutions/3747860/python-java-c-simple-solution-easy-to-understand/

**Description:** Given the root of a binary tree, the value of a target node target, and an integer k, return an array of the values of all nodes that have a distance k from the target node. You can return the answer in any order.

**Constraints:** 
The number of nodes in the tree is in the range [1, 500].
0 <= Node.val <= 500
All the values Node.val are unique.
target is the value of one of the nodes in the tree.
0 <= k <= 1000

**Examples:** 

```python3
root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, k = 2 #=> [7,4,1]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/fff42e72-d1d7-48f3-a7c8-a85618be33a1)

```python3
root = [1], target = 1, k = 3 #=> []
```


**Hint:**
1. Build adjacency list representation of the binary tree
2. Perform level order BFS on the graph, keeping track of distance
3. Capture nodes that are distance K from target

```python3
class Solution:
    def distanceK(self, root, target, K):
        # Step 1: Build adjacency list graph
        graph = {}
        self.buildGraph(root, None, graph)

        # Step 2: Perform BFS from the target node
        queue = [(target, 0)]
        visited = set([target])
        result = []
        
        while queue:
            node, distance = queue.pop(0)
            
            if distance == K:
                result.append(node.val)
                
            if distance > K:
                break
            
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        return result
    
    def buildGraph(self, node, parent, graph):
        if not node:
            return
        
        if node not in graph:
            graph[node] = []
            
        if parent:
            graph[node].append(parent)
            graph[parent].append(node)
            
        self.buildGraph(node.left, node, graph)
        self.buildGraph(node.right, node, graph)
```
**Time:** O(n)
**Space:** O(n)

## 83. Serialize and Deserialize Binary Tree
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0297-serialize-and-deserialize-binary-tree.py

**Description:** Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.

**Constraints:** 
The number of nodes in the tree is in the range [0, 10^4].
-1000 <= Node.val <= 1000

**Examples:** 

```python3
root = [1,2,3,null,null,4,5] #=> [1,2,3,null,null,4,5]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/90cbca67-10ae-47b9-8228-1c1233712625)

```python3
root = [] #=> []
```

**Hint:**
1. Serialize: traverse DFS creating a string separated by commas, if null node use char to identify.
2. Deserialize: split(",") then DFS creating new nodes

```python3
class Codec:
    def serialize(self, root):
        res = []

        def dfs(node):
            if not node:
                res.append("N")
                return
            res.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ",".join(res)

    def deserialize(self, data):
        vals = data.split(",")
        self.i = 0

        def dfs():
            if vals[self.i] == "N":
                self.i += 1
                return None
            node = TreeNode(int(vals[self.i]))
            self.i += 1
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()
```
**Time:** O(n)
**Space:** O(n)

## 84. Binary Tree Maximum Path Sum
**Reference:** https://leetcode.com/problems/binary-tree-maximum-path-sum/solutions/419793/python-recursive-solution-beats-98-in-time-and-75-in-memory/

**Description:** A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.

The path sum of a path is the sum of the node's values in the path.

Given the root of a binary tree, return the maximum path sum of any non-empty path.

**Constraints:** 
The number of nodes in the tree is in the range [1, 3 * 10^4].
-1000 <= Node.val <= 1000

**Examples:** 

```python3
root = [1,2,3] #=> 6
```

![image](https://github.com/will4skill/algo-review/assets/10373005/472d99f9-5cac-4b00-bece-8da7ee8cddf4)

```python3
root = [-10,9,20,null,null,15,7] #=> 42
```

![image](https://github.com/will4skill/algo-review/assets/10373005/00ba8c51-9752-4a58-a24b-de4198853b82)


**Hint:** Similar to diameter, but when you calculated left and right max, if they are < 0, return 0. When updating global, include current node's value.

```python3
class Solution:
    def maxPathSum(self, root):
        self.res = float('-inf')
        self.helper(root)
        return self.res 
        
    def helper(self, root):
        if not root:
            return 0
        left, right = self.helper(root.left), self.helper(root.right)
        self.res = max(self.res, root.val + left + right)
        return max(root.val + max(left, right), 0)
```
**Time:** O(n)
**Space:** O(n)

## 85. Binary Search
**Reference:** https://leetcode.com/problems/binary-search/solutions/1322419/5-variations-of-binary-search-a-self-note/

**Description:** Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1. You must write an algorithm with O(log n) runtime complexity.

**Constraints:** 
1 <= nums.length <= 10^4
-10^4 < nums[i], target < 10^4
All the integers in nums are unique.
nums is sorted in ascending order.

**Examples:** 

```python3
nums = [-1,0,3,5,9,12], target = 9 #=> 4
nums = [-1,0,3,5,9,12], target = 2 #=> -1
```

**Hint:** Just remember to round down. midIdx = startIdx + (endIdx - startIdx) // 2

```python3
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        startIdx, endIdx = 0, len(nums) - 1

        while startIdx <= endIdx:
            midIdx = startIdx + (endIdx - startIdx) // 2
            res = nums[midIdx]
            if res == target:
                return midIdx
            elif res > target:
                endIdx = midIdx - 1
            else:
                startIdx = midIdx + 1
        return -1
```
**Time:** O(log(n))
**Space:** O(1)

## 86. First Bad Version
**Reference:** https://leetcode.com/problems/first-bad-version/solutions/1591935/python-solution-easy-to-understand-binary-search-with-detailed-explanation/

**Description:** You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which returns whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

**Constraints:** 
1 <= bad <= n <= 2^31 - 1

**Examples:** 

```python3
n = 5, bad = 4 #=> 4
n = 1, bad = 1 #=> 1
```

**Hint:** Pretty generic binary search

```python3
class Solution:
    def firstBadVersion(self, n: int) -> int:
        start, end = 1, n
        while start < end:
            mid = start + (end - start) // 2
            if isBadVersion(mid):
                end = mid
            if not isBadVersion(mid):
                start = mid + 1
        return start
```
**Time:** O(log(n))
**Space:** O(1)

## 87. Search in Rotated Sorted Array
**Reference:** https://leetcode.com/problems/search-in-rotated-sorted-array/solutions/1786973/JavaScript-Solution/

**Description:** There is an integer array nums sorted in ascending order (with distinct values).

Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

You must write an algorithm with O(log n) runtime complexity.

**Constraints:** 
1 <= nums.length <= 5000
-10^4 <= nums[i] <= 10^4
All values of nums are unique.
nums is an ascending array that is possibly rotated.
-10^4 <= target <= 10^4

**Examples:** 

```python3
nums = [4,5,6,7,0,1,2], target = 0 #=> 4
nums = [4,5,6,7,0,1,2], target = 3 #=> -1
nums = [1], target = 0 #=> -1
```

**Hint:** Use binary search. Find lowest number (modified binary search with if and else only). Determine which side of min target is on. Use normal binary search on appropriate side of array. Reurn target idx or -1.

```python3
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # O(log n) Time Complexity
        # Use modified binary search to find lowest number index
        # Then, use regular binary search either to left or right of min depending on certain conditions

        if len(nums) == 0 or not nums:
            return -1

        left = 0
        right = len(nums) - 1

        # Modified binary search to find lowest num. While loop breaks out once left = right, smallest num is found
        while left < right:
            middle = (left + right) // 2
            if nums[middle] > nums[right]:
                left = middle + 1
            else:
                right = middle

        min_index = left
        left = 0
        right = len(nums) - 1

        # Now decide whether to search to left or right of min
        if target >= nums[min_index] and target <= nums[right]:
            left = min_index
        else:
            right = min_index - 1

        # Regular binary search
        while left <= right:
            middle = (left + right) // 2
            if target == nums[middle]:
                return middle
            elif target > nums[middle]:
                left = middle + 1
            else:
                right = middle - 1

        return -1
```
**Time:** O(log(n))
**Space:** O(1)

## 88. Time Based Key-Value Store
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0981-time-based-key-value-store.py

**Description:** Design a time-based key-value data structure that can store multiple values for the same key at different time stamps and retrieve the key's value at a certain timestamp.

Implement the TimeMap class:
1. TimeMap() Initializes the object of the data structure.
2. void set(String key, String value, int timestamp) Stores the key key with the value value at the given time timestamp.
3. String get(String key, int timestamp) Returns a value such that set was called previously, with timestamp_prev <= timestamp. If there are multiple such values, it returns the value associated with the largest timestamp_prev. If there are no values, it returns "".

**Constraints:** 
1 <= key.length, value.length <= 100
key and value consist of lowercase English letters and digits.
1 <= timestamp <= 10^7
All the timestamps timestamp of set are strictly increasing.
At most 2 * 10^5 calls will be made to set and get.

**Examples:** 

```python3
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]] #=> [null, null, "bar", "bar", null, "bar2", "bar2"]
```

**Hint:** 
1. set(): push (time stamp, value) for key
2. get(): find time <= timestamp using binary search. If no results, => empty string

```python3
class TimeMap:
    def __init__(self):
        self.keyStore = {}  # key : list of [val, timestamp]

    def set(self, key: str, value: str, timestamp: int) -> None:
        if key not in self.keyStore:
            self.keyStore[key] = []
        self.keyStore[key].append([value, timestamp])

    def get(self, key: str, timestamp: int) -> str:
        res, values = "", self.keyStore.get(key, [])
        l, r = 0, len(values) - 1
        while l <= r:
            m = (l + r) // 2
            if values[m][1] <= timestamp:
                res = values[m][0]
                l = m + 1
            else:
                r = m - 1
        return res
```
**Time:** O(n), O(log n)
**Space:** O(n), O(1)

## 89. Search a 2D Matrix (rows are sorted)
**Reference:** https://leetcode.com/problems/search-a-2d-matrix/solutions/274992/search-in-2d-matrix/

**Description:** You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.

**Constraints:** 
m == matrix.length
n == matrix[i].length
1 <= m, n <= 100
-10^4 <= matrix[i][j], target <= 10^4

**Examples:** 

```python3
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3 #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/55a7746d-f475-4cfe-a9f0-8b89ed14bdc3)

```python3
matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 13 #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/5297b3ba-df5a-444c-885e-3b91e8785bd9)

**Hint:** Sorted like an S. Use normal binary search except: midValue = Matrix[Math.floor(midIdx/cols)][midIdx%cols]

```javascript
const searchMatrix = (matrix, target) => {
  const rows = matrix.length;
  const cols = matrix[0].length;

  // Binary Search
  let startIdx = 0, endIdx = rows*cols - 1;
  let midIdx, midValue;

  while (startIdx <= endIdx) {
    midIdx = startIdx + Math.floor((endIdx - startIdx)/2);
    midValue = matrix[Math.floor(midIdx / cols)][midIdx % cols];
    if (target === midValue)
      return true
    else if (target < midValue)
      endIdx = midIdx - 1;
    else
      startIdx = midIdx + 1;
  }
  return false;
};
```
**Time:** O(log(m*n))
**Space:** O(1)

## 90. Find Minimum in Rotated Sorted Array
**Reference:** https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/solutions/158940/beat-100-very-simple-python-very-detailed-explanation/

**Description:** Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

1. [4,5,6,7,0,1,2] if it was rotated 4 times.
2. [0,1,2,4,5,6,7] if it was rotated 7 times.
Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

Given the sorted rotated array nums of unique elements, return the minimum element of this array.

You must write an algorithm that runs in O(log n) time.

**Constraints:** 
n == nums.length
1 <= n <= 5000
-5000 <= nums[i] <= 5000
All the integers of nums are unique.
nums is sorted and rotated between 1 and n times.

**Examples:** 

```python3
nums = [3,4,5,1,2] #=> 1
nums = [4,5,6,7,0,1,2] #=> 0
nums = [11,13,15,17] #=> 11
```

**Hint:** Use modified binary search.

```python3
class Solution:
    def findMin(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # set left and right bounds
        left, right = 0, len(nums)-1
                
        # left and right both converge to the minimum index;
        # DO NOT use left <= right because that would loop forever
        while left < right:
            # find the middle value between the left and right bounds (their average);
			# can equivalently do: mid = left + (right - left) // 2,
			# if we are concerned left + right would cause overflow (which would occur
			# if we are searching a massive array using a language like Java or C that has
			# fixed size integer types)
            mid = (left + right) // 2
                        
            # the main idea for our checks is to converge the left and right bounds on the start
            # of the pivot, and never disqualify the index for a possible minimum value.

            # in normal binary search, we have a target to match exactly,
            # and would have a specific branch for if nums[mid] == target.
            # we do not have a specific target here, so we just have simple if/else.
                        
            if nums[mid] > nums[right]:
                # we KNOW the pivot must be to the right of the middle:
                # if nums[mid] > nums[right], we KNOW that the
                # pivot/minimum value must have occurred somewhere to the right
                # of mid, which is why the values wrapped around and became smaller.

                # example:  [3,4,5,6,7,8,9,1,2] 
                # in the first iteration, when we start with mid index = 4, right index = 9.
                # if nums[mid] > nums[right], we know that at some point to the right of mid,
                # the pivot must have occurred, which is why the values wrapped around
                # so that nums[right] is less then nums[mid]

                # we know that the number at mid is greater than at least
                # one number to the right, so we can use mid + 1 and
                # never consider mid again; we know there is at least
                # one value smaller than it on the right
                left = mid + 1

            else:
                # here, nums[mid] <= nums[right]:
                # we KNOW the pivot must be at mid or to the left of mid:
                # if nums[mid] <= nums[right], we KNOW that the pivot was not encountered
                # to the right of middle, because that means the values would wrap around
                # and become smaller (which is caught in the above if statement).
                # this leaves the possible pivot point to be at index <= mid.
                            
                # example: [8,9,1,2,3,4,5,6,7]
                # in the first iteration, when we start with mid index = 4, right index = 9.
                # if nums[mid] <= nums[right], we know the numbers continued increasing to
                # the right of mid, so they never reached the pivot and wrapped around.
                # therefore, we know the pivot must be at index <= mid.

                # we know that nums[mid] <= nums[right].
                # therefore, we know it is possible for the mid index to store a smaller
                # value than at least one other index in the list (at right), so we do
                # not discard it by doing right = mid - 1. it still might have the minimum value.
                right = mid
                
        # at this point, left and right converge to a single index (for minimum value) since
        # our if/else forces the bounds of left/right to shrink each iteration:

        # when left bound increases, it does not disqualify a value
        # that could be smaller than something else (we know nums[mid] > nums[right],
        # so nums[right] wins and we ignore mid and everything to the left of mid).

        # when right bound decreases, it also does not disqualify a
        # value that could be smaller than something else (we know nums[mid] <= nums[right],
        # so nums[mid] wins and we keep it for now).

        # so we shrink the left/right bounds to one value,
        # without ever disqualifying a possible minimum
        return nums[left]
```
**Time:** O(log(n))
**Space:** O(1)

## 91. Maximum Profit in Job Scheduling
**Reference:** https://leetcode.com/problems/maximum-profit-in-job-scheduling/solutions/409009/java-c-python-dp-solution/

**Description:** We have n jobs, where every job is scheduled to be done from startTime[i] to endTime[i], obtaining a profit of profit[i].

You're given the startTime, endTime and profit arrays, return the maximum profit you can take such that there are no two jobs in the subset with overlapping time range.

If you choose a job that ends at time X you will be able to start another job that starts at time X.

**Constraints:** 
1 <= startTime.length == endTime.length == profit.length <= 5 * 10^4
1 <= startTime[i] < endTime[i] <= 10^9
1 <= profit[i] <= 10^4

**Examples:** 

```python3
startTime = [1,2,3,3], endTime = [3,4,5,6], profit = [50,10,40,70] #=> 120
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d91a37ac-e8f7-4f39-94b7-0fedb9ef09e5)

```python3
startTime = [1,2,3,4,6], endTime = [3,5,10,6,9], profit = [20,20,100,70,60] #=> 150
```

![image](https://github.com/will4skill/algo-review/assets/10373005/b4542813-9f7d-4b94-bd6c-d532275e9780)

```python3
startTime = [1,1,1], endTime = [2,3,4], profit = [5,6,4] #=> 6
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d483ac47-d253-4a13-a6d8-02d8e8655d2d)

**Hint:** Binary search and DP. 
1. Sort jobs by endTime
2. Use dp list to memoize dp[time] => maxprofit
3. Choice is similar to knapsack:
   a. Skip the job
   b. Take the job and binary search the dp to find the largest profit we can make before start time s
4. Compare the max current profit with the last element in dp. If curr profit is better, add new pair [e, cur] to back of dp

```python3
    def jobScheduling(self, startTime, endTime, profit):
        jobs = sorted(zip(startTime, endTime, profit), key=lambda v: v[1])
        dp = [[0, 0]]
        for s, e, p in jobs:
            i = bisect.bisect(dp, [s + 1]) - 1
            if dp[i][1] + p > dp[-1][1]:
                dp.append([e, dp[i][1] + p])
        return dp[-1][1]
```
**Time:** O(nlog n)
**Space:** O(n)

## 92. Median of Two Sorted Arrays
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0004-median-of-two-sorted-arrays.py

**Description:** Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

**Constraints:** 
nums1.length == m
nums2.length == n
0 <= m <= 1000
0 <= n <= 1000
1 <= m + n <= 2000
-10^6 <= nums1[i], nums2[i] <= 10^6

**Examples:** 
```python3
nums1 = [1,3], nums2 = [2] #=> 2.00000
nums1 = [1,2], nums2 = [3,4] #=> 2.50000
```

**Hint:** You can't just merge the two sorted arrays (too slow). You need to partition the array into two equal sizes. Try to find the combiined left partition from both arrays. Success if the last element in AL is <= the first element in BR and vice versa (mid is next or average of last and next). 

If the partition is incorrect, the array left pointer of the array who's endpoint is too small becomes mid + 1, then try again. Essentially, you make the too small (value not length) array shrink (move its right ptr back). Out of bounds to left = -infinity, to right = +infinity. p

```python3
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        A, B = nums1, nums2
        total = len(nums1) + len(nums2)
        half = total // 2

        if len(B) < len(A):
            A, B = B, A

        l, r = 0, len(A) - 1
        while True:
            i = (l + r) // 2  # A
            j = half - i - 2  # B

            Aleft = A[i] if i >= 0 else float("-infinity")
            Aright = A[i + 1] if (i + 1) < len(A) else float("infinity")
            Bleft = B[j] if j >= 0 else float("-infinity")
            Bright = B[j + 1] if (j + 1) < len(B) else float("infinity")

            # partition is correct
            if Aleft <= Bright and Bleft <= Aright:
                # odd
                if total % 2:
                    return min(Aright, Bright)
                # even
                return (max(Aleft, Bleft) + min(Aright, Bright)) / 2
            elif Aleft > Bright:
                r = i - 1
            else:
                l = i + 1
```
**Time:** O(log(min(n, m)))
**Space:** O(n)

## 93. flood-fill
**Reference:** https://leetcode.com/problems/flood-fill/solutions/2669996/dfs-bfs-solutions-explained-iterative-recursive/

**Description:** An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.

You are also given three integers sr, sc, and color. You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with color.

Return the modified image after performing the flood fill.

**Constraints:** 
m == image.length
n == image[i].length
1 <= m, n <= 50
0 <= image[i][j], color < 2^16
0 <= sr < m
0 <= sc < n

**Examples:** 
```python3
image = [[1,1,1],[1,1,0],[1,0,1]], sr = 1, sc = 1, color = 2 #=> [[2,2,2],[2,2,0],[2,0,1]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/759943f0-3562-4d88-83fb-221ebc7722e5)

```python3
image = [[0,0,0],[0,0,0]], sr = 0, sc = 0, color = 0 #=> [[0,0,0],[0,0,0]]
```

**Hint:** Visited = new color squares. BFS or DFS possible

```python3
# DFS
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
	start_color = image[sr][sc] #keep track of original color
	if start_color == color: #the color is already this one so do nothing
		return image
	m = len(image) #length of image
	n = len(image[0]) #column length

	def dfs(r, c): #dfs helper method
		image[r][c] = color #set spot to color
		for (row, col) in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]: #look at 4-dimensionally adjacent places
			if 0 <= row < m and 0 <= col < n and image[row][col] == start_color: #check if in bounds and equal to start_color
				dfs(row, col) #if so, we must search here
	dfs(sr, sc) #start searching at (sr,sc)
	return image
```

```python3
# BFS
def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
	start_color = image[sr][sc] #keep track of original color
	if start_color == color: #the color is already this one so do nothing
		return image
	m = len(image) #length of image
	n = len(image[0]) #column length
	
	queue = collections.deque([(sr, sc)]) #queue to keep track of spaces we need to look at
	while queue: #while there are places to look at
		(r, c) = queue.popleft() #get the next spot
		image[r][c] = color #set it to color
		for (row, col) in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]: #look at 4-dimensionally adjacent places
			if 0 <= row < m and 0 <= col < n and image[row][col] == start_color: #check if in bounds and equal to start_color
				queue.append((row, col)) #if so, we must search this place too
	return image
```
**Time:** O(n)
**Space:** O(n)

## 94. 01 Matrix
**Reference:** https://leetcode.com/problems/01-matrix/solutions/1369741/c-java-python-bfs-dp-solutions-with-picture-clean-concise-o-1-space/

**Description:** Given an m x n binary matrix mat, return the distance of the nearest 0 for each cell. The distance between two adjacent cells is 1.

**Constraints:** 
m == mat.length
n == mat[i].length
1 <= m, n <= 10^4
1 <= m * n <= 10^4
mat[i][j] is either 0 or 1.
There is at least one 0 in mat.

**Examples:** 

```python3
mat = [[0,0,0],[0,1,0],[0,0,0]] #=> [[0,0,0],[0,1,0],[0,0,0]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/413f2462-76c6-458e-9d06-9ec5e6ebe1d3)

```python3
mat = [[0,0,0],[0,1,0],[1,1,1]]] #=> [[0,0,0],[0,1,0],[1,2,1]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/23d988db-138c-46d5-88d5-1d4fe4ea1978)

**Hint:** Use BFS. Firstly, we can see that the distance of all zero-cells are 0.
Same idea with Topology Sort, we process zero-cells first, then we use queue data structure to keep the order of processing cells, so that cells which have the smaller distance will be processed first. Then we expand the unprocessed neighbors of the current processing cell and push into our queue.

```python3
# BFS Solution
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        DIR = [0, 1, 0, -1, 0]

        q = deque([])
        for r in range(m):
            for c in range(n):
                if mat[r][c] == 0:
                    q.append((r, c))
                else:
                    mat[r][c] = -1  # Marked as not processed yet!

        while q:
            r, c = q.popleft()
            for i in range(4):
                nr, nc = r + DIR[i], c + DIR[i + 1]
                if nr < 0 or nr == m or nc < 0 or nc == n or mat[nr][nc] != -1: continue
                mat[nr][nc] = mat[r][c] + 1
                q.append((nr, nc))
        return mat
```
**Time:** O(M * N), where M is number of rows, N is number of columns in the matrix.
**Space:** O(n)

![image](https://github.com/will4skill/algo-review/assets/10373005/e87d15c0-8893-4a82-aca2-3e3f55e17876)

```python3
# DP Solution
class Solution:  # 520 ms, faster than 96.50%
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])

        for r in range(m):
            for c in range(n):
                if mat[r][c] > 0:
                    top = mat[r - 1][c] if r > 0 else math.inf
                    left = mat[r][c - 1] if c > 0 else math.inf
                    mat[r][c] = min(top, left) + 1

        for r in range(m - 1, -1, -1):
            for c in range(n - 1, -1, -1):
                if mat[r][c] > 0:
                    bottom = mat[r + 1][c] if r < m - 1 else math.inf
                    right = mat[r][c + 1] if c < n - 1 else math.inf
                    mat[r][c] = min(mat[r][c], bottom + 1, right + 1)
        return mat
```
**Time:** O(M * N), where M is number of rows, N is number of columns in the matrix.
**Space:** O(1)

## 95. Clone Graph
**Reference:** https://leetcode.com/problems/clone-graph/solutions/1404781/python-easy-clean-code/

**Description:** Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.
```java
class Node {
    public int val;
    public List<Node> neighbors;
}
``` 

Test case format:

For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.

An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.

The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

**Constraints:** 
The number of nodes in the graph is in the range [0, 100].
1 <= Node.val <= 100
Node.val is unique for each node.
There are no repeated edges and no self-loops in the graph.
The Graph is connected and all nodes can be visited starting from the given node.

**Examples:** 
```python3
adjList = [[2,4],[1,3],[2,4],[1,3]] #=> [[2,4],[1,3],[2,4],[1,3]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/29919501-bcc2-46da-b5b7-7941a81e922b)

```python3
adjList = [[]] #=> [[]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/cfa5ff05-7b45-4c17-855b-c8a1fb05458a)

```python3
adjList = [] #=> []
```

**Hint:** Visited = hashMap where node => newNode. Create new neighbors of clone as you iterate.

```python3
class Solution:
    def helper(self, node, visited):
        if node is None:
            return None
        
        newNode = Node(node.val)
        visited[node.val] = newNode
        
        for adjNode in node.neighbors:
            if adjNode.val not in visited:
                newNode.neighbors.append(self.helper(adjNode, visited))
            else:
                newNode.neighbors.append(visited[adjNode.val])
        
        return newNode
    
    def cloneGraph(self, node: 'Node') -> 'Node':
        return self.helper(node, {})
```
**Time:** O(n + m): n = nodes, m = edges
**Space:** O(n): space in visited

## 96. Course Schedule
**Reference:** https://leetcode.com/problems/course-schedule/solutions/58586/python-20-lines-dfs-solution-sharing-with-explanation/

**Description:** There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.

**Constraints:** 
1 <= numCourses <= 2000
0 <= prerequisites.length <= 5000
prerequisites[i].length == 2
0 <= ai, bi < numCourses
All the pairs prerequisites[i] are unique.

**Examples:** 

```python3
numCourses = 2, prerequisites = [[1,0]] #=> true
numCourses = 2, prerequisites = [[1,0],[0,1]] #=> false
```

**Hint:** Topological sort. See 44. prereqs possible. Same as cycle detect, but you have to create adj list. Cycles result in a false return.

```python3
class Solution(object):
    def canFinish(self, numCourses, prerequisites):
        graph = [[] for _ in range(numCourses)]
        visited = [0 for _ in range(numCourses)]
        # create graph
        for pair in prerequisites:
            x, y = pair
            graph[x].append(y)
        # visit each node
        for i in range(numCourses):
            if not self.dfs(graph, visited, i):
                return False
        return True
    
    def dfs(self, graph, visited, i):
        # if ith node is marked as being visited, then a cycle is found
        if visited[i] == -1:
            return False
        # if it is done visted, then do not visit again
        if visited[i] == 1:
            return True
        # mark as being visited
        visited[i] = -1
        # visit all the neighbours
        for j in graph[i]:
            if not self.dfs(graph, visited, j):
                return False
        # after visit all the neighbours, mark it as done visited
        visited[i] = 1
        return True
```
**Time:** O(n + p) p = # prereqs n = # courses
**Space:** O(n)

## 97. Number of Islands
**Reference:** https://www.structy.net/problems/island-count

**Description:** Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.
An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

**Constraints:** 
m == grid.length
n == grid[i].length
1 <= m, n <= 300
grid[i][j] is '0' or '1'.

**Examples:** 
```python3
grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
#=> 1

grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]
#=> 3
```

**Hint:** See 37. Attempt to enter each grid[i][j]. If you can explore, count++. Make sure water "w" is a base case => False
Note: you could make space O(1) by using grid as visited set.

```python3
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m,n = len(grid),len(grid[0])
        visited = set()
        count = 0
        for r in range(m):
            for c in range(n):
                if self.explore(grid, r, c, visited) == True:
                    count += 1
        return count

    def explore(self, grid, r, c, visited):
        row_inbounds = 0 <= r < len(grid)
        col_inbounds = 0 <= c < len(grid[0])
        if not row_inbounds or not col_inbounds:
            return False
        
        if grid[r][c] == '0':
            return False
        
        pos = (r, c)
        if pos in visited:
            return False
        visited.add(pos)
        
        self.explore(grid, r - 1, c, visited)
        self.explore(grid, r + 1, c, visited)  
        self.explore(grid, r, c - 1, visited)
        self.explore(grid, r, c + 1, visited)
        
        return True
```
**Time:** Time: O(rc) r = number of rows c = number of columns
**Space:** O(rc)

## 98. Rotting Oranges
**Reference:** https://leetcode.com/problems/rotting-oranges/solutions/563686/python-clean-well-explained-faster-than-90/

**Description:** You are given an m x n grid where each cell can have one of three values:

1. 0 representing an empty cell,
2. 1 representing a fresh orange, or
3. 2 representing a rotten orange.

Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.

**Constraints:** 
m == grid.length
n == grid[i].length
1 <= m, n <= 10
grid[i][j] is 0, 1, or 2.

**Examples:** 
```python3
grid = [[2,1,1],[1,1,0],[0,1,1]] #=> 4
```

![image](https://github.com/will4skill/algo-review/assets/10373005/1b1ac9b4-e72c-4034-a05d-0e0b412f86a8)

```python3
grid = [[2,1,1],[0,1,1],[1,0,1]] #=> -1
```

```python3
grid = [[0,2]] #=> 0
```

**Hint:** BFS. Traverse the graph and load up all the rotting oranges into your queue. While the queue is not empty, explore all appropriate neighbors. Increment time each iteration. Update orange state as necessary.

```python3
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        # number of rows
        rows = len(grid)
        if rows == 0:  # check if grid is empty
            return -1
        # number of columns
        cols = len(grid[0])
        # keep track of fresh oranges
        fresh_cnt = 0
        # queue with rotten oranges (for BFS)
        rotten = deque()
        # visit each cell in the grid
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    # add the rotten orange coordinates to the queue
                    rotten.append((r, c))
                elif grid[r][c] == 1:
                    # update fresh oranges count
                    fresh_cnt += 1
        
        # keep track of minutes passed.
        minutes_passed = 0
        # If there are rotten oranges in the queue and there are still fresh oranges in the grid keep looping
        while rotten and fresh_cnt > 0:
            # update the number of minutes passed
            # it is safe to update the minutes by 1, since we visit oranges level by level in BFS traversal.
            minutes_passed += 1
            # process rotten oranges on the current level
            for _ in range(len(rotten)):
                x, y = rotten.popleft()
                # visit all the adjacent cells
                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    # calculate the coordinates of the adjacent cell
                    xx, yy = x + dx, y + dy
                    # ignore the cell if it is out of the grid boundary
                    if xx < 0 or xx == rows or yy < 0 or yy == cols:
                        continue
                    # ignore the cell if it is empty '0' or visited before '2'
                    if grid[xx][yy] == 0 or grid[xx][yy] == 2:
                        continue
                    # update the fresh oranges count
                    fresh_cnt -= 1
                    # mark the current fresh orange as rotten
                    grid[xx][yy] = 2
                    # add the current rotten to the queue
                    rotten.append((xx, yy))
        # return the number of minutes taken to make all the fresh oranges to be rotten
        # return -1 if there are fresh oranges left in the grid (there were no adjacent rotten oranges to make them rotten)
        return minutes_passed if fresh_cnt == 0 else -1
```
**Time:** O(rows * cols)
**Space:** O(rows * cols)

## 99. Accounts Merge
**Reference:** https://leetcode.com/problems/accounts-merge/solutions/109161/python-simple-dfs-with-explanation/

**Description:** Given a list of accounts where each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some common email to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

**Constraints:** 
1 <= accounts.length <= 1000
2 <= accounts[i].length <= 10
1 <= accounts[i][j].length <= 30
accounts[i][0] consists of English letters.
accounts[i][j] (for j > 0) is a valid email.

**Examples:** 
```python3
accounts = [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
#=> [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]

accounts = [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
#=> [["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]
```

**Hint:** DFS. Create a map where key is the email and value is a list of accounts you can find it in. This is our graph. Perform DFS on each account in the list while using the map to tell us which accounts are linked to that particular account via common emails. This will make sure we visit each account only once. Collect the emails as you visit them. Sort the resulting collection.

```python3
class Solution(object):
    def accountsMerge(self, accounts):
        from collections import defaultdict
        visited_accounts = [False] * len(accounts)
        emails_accounts_map = defaultdict(list)
        res = []
        # Build up the graph.
        for i, account in enumerate(accounts):
            for j in range(1, len(account)):
                email = account[j]
                emails_accounts_map[email].append(i)
        # DFS code for traversing accounts.
        def dfs(i, emails):
            if visited_accounts[i]:
                return
            visited_accounts[i] = True
            for j in range(1, len(accounts[i])):
                email = accounts[i][j]
                emails.add(email)
                for neighbor in emails_accounts_map[email]:
                    dfs(neighbor, emails)
        # Perform DFS for accounts and add to results.
        for i, account in enumerate(accounts):
            if visited_accounts[i]:
                continue
            name, emails = account[0], set()
            dfs(i, emails)
            res.append([name] + sorted(emails))
        return res
```
**Time:** O(n*logn)
**Space:** O(n) ??

## 100. Word Search
**Reference:** https://leetcode.com/problems/word-search/solutions/27660/python-dfs-solution-with-comments/

**Description:** Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

Follow up: Could you use search pruning to make your solution faster with a larger board?

**Constraints:** 
m == board.length
n = board[i].length
1 <= m, n <= 6
1 <= word.length <= 15
board and word consists of only lowercase and uppercase English letters.

**Examples:** 
```python3
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED" #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/86b824c0-bcad-4f6f-bf10-a91ceccb0dcb)

```python3
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "SEE" #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/67ab7282-1f10-46fe-8e52-7164e700d631)


```python3
board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCB" #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/2eaf6c98-df03-4527-a27b-733aef27ba9c)


**Hint:** Standard DFS starting at each character and trying to complete word. Similar to sum problems, remove found character and then recurse until False or len(word) == 0

```python3
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        height, width = len(board), len(board[0])
        visited = set()
        if height < 1 or width < 1: return False
        for i in range(height):
            for j in range(width):
                if self.existHelper(board, word, (i, j), 0, visited) == True:
                    return True
        return False

    def existHelper(self, board, word, location, index, visited):
        if index == len(word): 
            return True
        height, width = len(board), len(board[0])
        if location[0] < 0 or location[1] < 0 or location[0] >= height or location[1] >= width:
            return False
        if location in visited or board[location[0]][location[1]] != word[index]:
            return False

        visited.add(location) # Tim note: 🤯
        result = (self.existHelper(board, word, (location[0] - 1, location[1]), index + 1, visited) or
        self.existHelper(board, word, (location[0] + 1, location[1]), index + 1, visited) or
        self.existHelper(board, word, (location[0], location[1] + 1), index + 1, visited) or
        self.existHelper(board, word, (location[0], location[1] - 1), index + 1, visited))
        visited.remove(location) # Tim note: 🤯

        return result
```
**Time:** O(m*n4^s) where m=# of rows, n=# of cols and s=len of the word.
**Space:** ???

## 101. Minimum Height Trees
**Reference:** https://leetcode.com/problems/minimum-height-trees/solutions/76055/share-some-thoughts/

**Description:** A tree is an undirected graph in which any two vertices are connected by exactly one path. In other words, any connected graph without simple cycles is a tree.

Given a tree of n nodes labelled from 0 to n - 1, and an array of n - 1 edges where edges[i] = [ai, bi] indicates that there is an undirected edge between the two nodes ai and bi in the tree, you can choose any node of the tree as the root. When you select a node x as the root, the result tree has height h. Among all possible rooted trees, those with minimum height (i.e. min(h))  are called minimum height trees (MHTs).

Return a list of all MHTs' root labels. You can return the answer in any order.

The height of a rooted tree is the number of edges on the longest downward path between the root and a leaf.

**Constraints:** 
1 <= n <= 2 * 10^4
edges.length == n - 1
0 <= ai, bi < n
ai != bi
All the pairs (ai, bi) are distinct.
The given input is guaranteed to be a tree and there will be no repeated edges.

**Examples:** 

```python3
n = 4, edges = [[1,0],[1,2],[1,3]] #=> [1]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/e51c4d75-187a-4ee2-b4e6-6812e18b0ec1)

```python3
n = 6, edges = [[3,0],[3,1],[3,2],[3,4],[5,4]] #=> [3,4]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/81aa38e9-6eec-47a3-ac7b-86946bf14858)

**Hint:** Convert edge list to adj matrix. Select all the leaves. While more than two nodes are left in graph, remove current layer of leaves. Return remaining 1 or 2 nodes.

```python3
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
        if n == 1: return [0] 
        adj = [set() for _ in range(n)]
        for i, j in edges:
            adj[i].add(j)
            adj[j].add(i)

        leaves = [i for i in range(n) if len(adj[i]) == 1]

        while n > 2:
            n -= len(leaves)
            newLeaves = []
            for i in leaves:
                j = adj[i].pop()
                adj[j].remove(i)
                if len(adj[j]) == 1: newLeaves.append(j)
            leaves = newLeaves
        return leaves   
```
**Time:** O(n)
**Space:** O(n)

## 102. Pacific Atlantic Water Flow
**Reference:** https://leetcode.com/problems/pacific-atlantic-water-flow/solutions/1126938/short-easy-w-explanation-diagrams-simple-graph-traversals-dfs-bfs/

**Description:** There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

**Constraints:** 
m == heights.length
n == heights[r].length
1 <= m, n <= 200
0 <= heights[r][c] <= 10^5

**Examples:** 

```python3
heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]] #=> [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/93f98937-f9a3-4324-955a-fb10ffab4fe8)

```python3
heights = [[1]] #=> [[0,0]]
```

**Hint:** BFS: start with all cells adjacent to one of the oceans (A). Visited neighbors that are greater than or equal to the starting nodes until you reach a subset of cells adjacent to the other ocean (B). Do the same from B to A. The final answer we get will be the intersection of sets A and B (A ∩ B). 

```cpp
class Solution {
public:
    int m, n;
    vector<vector<int> > ans;
    vector<vector<bool> > atlantic, pacific;
    queue<pair<int, int> > q;
    vector<vector<int> > pacificAtlantic(vector<vector<int>>& mat) {
        if(!size(mat)) return ans;
        m = size(mat), n = size(mat[0]);
        atlantic = pacific = vector<vector<bool> >(m, vector<bool>(n, false));
        for(int i = 0; i < m; i++) bfs(mat, pacific, i, 0), bfs(mat, atlantic, i, n - 1);
        for(int i = 0; i < n; i++) bfs(mat, pacific, 0, i), bfs(mat, atlantic, m - 1, i);             
        return ans;
    }
    void bfs(vector<vector<int> >& mat, vector<vector<bool> >& visited, int i, int j){        
        q.push({i, j});
        while(!q.empty()){
            tie(i, j) = q.front(); q.pop();
            if(visited[i][j]) continue;
            visited[i][j] = true;
            if(atlantic[i][j] && pacific[i][j]) ans.push_back(vector<int>{i, j});
            if(i + 1 <  m && mat[i + 1][j] >= mat[i][j]) q.push({i + 1, j});
            if(i - 1 >= 0 && mat[i - 1][j] >= mat[i][j]) q.push({i - 1, j});
            if(j + 1 <  n && mat[i][j + 1] >= mat[i][j]) q.push({i, j + 1});
            if(j - 1 >= 0 && mat[i][j - 1] >= mat[i][j]) q.push({i, j - 1});
        }
    }
};
```
**Time:** O(M*N)
**Space:** O(M*N)

## 103. Shortest Path to Get Food
**Reference:** https://www.cnblogs.com/cnoodle/p/15645191.html

**Description:** You are starving and you want to eat food as quickly as possible. You want to find the shortest path to arrive at any food cell.

You are given an m x n character matrix, grid, of these different types of cells:

1. '*' is your location. There is exactly one '*' cell.
2. '#' is a food cell. There may be multiple food cells.
3. 'O' is free space, and you can travel through these cells.
4. 'X' is an obstacle, and you cannot travel through these cells.
You can travel to any adjacent cell north, east, south, or west of your current location if there is not an obstacle.

Return the length of the shortest path for you to reach any food cell. If there is no path for you to reach food, return -1.

**Constraints:** 
m == grid.length
n == grid[i].length
1 <= m, n <= 200
grid[row][col] is '*', 'X', 'O', or '#'.
The grid contains exactly one '*'.

**Examples:** 

```python3
grid = [["X","X","X","X","X","X"],["X","*","O","O","O","X"],["X","O","O","#","O","X"],["X","X","X","X","X","X"]] #=> [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]] #=> 3
```

![image](https://github.com/will4skill/algo-review/assets/10373005/53520692-5cd8-4bfb-91a7-b498f12abf4d)


```python3
grid = [["X","X","X","X","X"],["X","*","X","O","X"],["X","O","X","#","X"],["X","X","X","X","X"]] #=> -1
```

![image](https://github.com/will4skill/algo-review/assets/10373005/2c257507-4d12-4f1a-92d2-0a9653186200)


```python3
grid = [["X","X","X","X","X","X","X","X"],["X","*","O","X","O","#","O","X"],["X","O","O","X","O","O","X","X"],["X","O","O","O","O","#","O","X"],["X","X","X","X","X","X","X","X"]] #=> 6
```

![image](https://github.com/will4skill/algo-review/assets/10373005/6743b25b-fb5b-4235-a481-2be37fe5afc2)

```python3
grid = [["O","*"],["#","O"]] #=> 2
grid = [["X","*"],["#","X"]] #=> -1
```

**Hint:** See 39. Closest carrot. Essentially, grid graph shortest path. BFS (+= 1 for each new layer)

```java
class Solution {
    int[][] DIRS = { { 0, -1 }, { 0, 1 }, { 1, 0 }, { -1, 0 } };

    public int getFood(char[][] grid) {
        int m = grid.length;
        int n = grid[0].length;
        Queue<int[]> queue = new LinkedList<>();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == '*') {
                    queue.offer(new int[] { i, j });
                    break;
                }
            }
        }
        boolean[][] visited = new boolean[m][n];

        int step = 0;
        while (!queue.isEmpty()) {
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                int[] cur = queue.poll();
                int x = cur[0];
                int y = cur[1];
                if (grid[x][y] == '#') {
                    return step;
                }
                for (int[] dir : DIRS) {
                    int r = x + dir[0];
                    int c = y + dir[1];
                    if (r >= 0 && r < m && c >= 0 && c < n && grid[r][c] != 'X' && !visited[r][c]) {
                        visited[r][c] = true;
                        queue.offer(new int[] { r, c });
                    }
                }
            }
            step++;
        }
        return -1;
    }
}
```
**Time:** O(rc) r = number of rows c = number of columns
**Space:** O(rc)

## 104. Graph Valid Tree
**Reference:** https://algomonster.medium.com/leetcode-261-graph-valid-tree-f27c212c1db1

**Description:** Given n nodes labeled from 0 to n-1 and a list of undirected edges (each edge is a pair of nodes), write a function to check whether these edges make up a valid tree.

**Constraints:** 
Note: you can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0,1] is the same as [1,0] and thus will not appear together in edges.

**Examples:** 

```python3
n = 5, and edges = [[0,1], [0,2], [0,3], [1,4]] #=> true
n = 5, and edges = [[0,1], [1,2], [2,3], [1,3], [1,4]] #=> false
```

**Hint:** A tree is a special undirected graph. It satisfies two properties: 
1. It is connected
2. It has no cycle.

Try Iterate over entire graph with dfs. If you visit all nodes, it is connected. See structy, visited/visiting for cycle check: 1 use two sets, visited and visiting (behaves like normal visited). 2. Test each node in the graph. 3. After all explored, remove from visiting, add to visited, and return False

```python3
# Standard Solution
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        from collections import defaultdict
        graph = defaultdict(list)
        
        # build the graph
        for src, dest in edges:
            graph[src].append(dest)
            graph[dest].append(src)
            
        visited = set()
        def dfs(root, parent): # returns true if graph has no cycle
            visited.add(root)
            for node in graph[root]:
                if node == parent: # trivial cycle, skip
                    continue
                if node in visited:
                    return False
            
                if not dfs(node, root):
                    return False
            return True
        
        return dfs(0, -1) and len(visited) == n
```

```python3
# Simplified Solution (no cycle if # of nodes == # of edges + 1)
class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        from collections import defaultdict
        graph = defaultdict(list)
        
        # build the graph
        for src, dest in edges:
            graph[src].append(dest)
            graph[dest].append(src)
            
        visited = set()f
        def dfs(root):
            visited.add(root)
            for node in graph[root]:
                if node in visited:
                    continue
                dfs(node)
            
        dfs(0)
        return len(visited) == n and len(edges) == n - 1
```
**Time:** O(n^2)
**Space:** O(n)

## 105. Course Schedule II
**Reference:** https://leetcode.com/problems/course-schedule-ii/submissions/1155573265/ 
vsharda1

**Description:** There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

**Constraints:** 
1 <= numCourses <= 2000
0 <= prerequisites.length <= numCourses * (numCourses - 1)
prerequisites[i].length == 2
0 <= ai, bi < numCourses
ai != bi
All the pairs [ai, bi] are distinct.

**Examples:** 

```python3
numCourses = 2, prerequisites = [[1,0]] #=> [0,1]
numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]] #=> [0,2,1,3]
numCourses = 1, prerequisites = [] #=> [0]
```

**Hint:** 
See structy, topological order. Convert the edges to an adjacency list. 
1. Use a map to track the number of parents each node has numParent[node] => #
2. Make a list of all source nodes (i.e., they don't have parents)
3. Use modified DFS to explore the source nodes whle adding new source nodes as their parents are removed

```python3
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        sortedorder = []
        if numCourses <= 0:
            return False
        inDegree = {i : 0 for i in range(numCourses)}
        graph = {i : [] for i in range(numCourses)}
        
        for child, parent in prerequisites:
            graph[parent].append(child)
            inDegree[child] += 1

        sources = deque()
        
        for key in inDegree:
            if inDegree[key] == 0:
                sources.append(key)
        #visited = 0       
        while sources:
            vertex = sources.popleft()
            #visited += 1
            sortedorder.append(vertex)
            for child in graph[vertex]:
                inDegree[child] -= 1
                if inDegree[child] == 0:
                    sources.append(child)
        
        if len(sortedorder) != numCourses:
            return []
        return sortedorder
```

**Time:** O(e + n) e = number of edges n = number of nodes
**Space:** O(n)

## 106. Number of Connected Components in an Undirected Graph
**Reference:** https://leetcode.ca/all/323.html

**Description:** Given n nodes labeled from 0 to n - 1 and a list of undirected edges (each edge is a pair of nodes), write a function to find the number of connected components in an undirected graph.

**Constraints:** 
You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.

**Examples:** 

```python3
n = 5 and edges = [[0, 1], [1, 2], [3, 4]] #=> 2
     0          3
     |          |
     1 --- 2    4

n = 5 and edges = [[0, 1], [1, 2], [2, 3], [3, 4]] #=> 1
     0           4
     |           |
     1 --- 2 --- 3
```

**Hint:** 
Create adjacency matrix. Explore each node, keeping a global visited. When you complete a new component, count += 1

```python3
# Structy Solution. Note: you have to create an adjacency matrix first
def connected_components_count(graph):
  visited = set()
  count = 0
  
  for node in graph:
    if explore(graph, node, visited) == True:
      count += 1
      
  return count

def explore(graph, current, visited):
  if current in visited:
    return False
  
  visited.add(current)
  
  for neighbor in graph[current]:
    explore(graph, neighbor, visited)
  
  return True

def build_graph(edges):
  graph = {}
  
  for edge in edges:
    a, b = edge
    
    if a not in graph:
      graph[a] = []
    if b not in graph:
      graph[b] = []
      
    graph[a].append(b)
    graph[b].append(a)
    
  return graph
```

**Time:** O(e) n = number of nodes e = number edges
**Space:** O(n)

## 107. Minimum Knight Moves
**Reference:** https://leetcode.ca/2019-03-11-1197-Minimum-Knight-Moves/

**Description:** In an infinite chess board with coordinates from -infinity to +infinity, you have a knight at square [0, 0].

A knight has 8 possible moves it can make, as illustrated below. Each move is two squares in a cardinal direction, then one square in an orthogonal direction.

![image](https://github.com/will4skill/algo-review/assets/10373005/2af1b994-0ee5-49f3-882d-83475edaf597)

Return the minimum number of steps needed to move the knight to the square [x, y]. It is guaranteed the answer exists.

**Constraints:** 
-300 <= x, y <= 300
0 <= |x| + |y| <= 300

**Examples:** 
```python3
x = 2, y = 1 #=> 1
x = 5, y = 5 #=> 4
```

**Hint:** 
Use BFS over the grid with the additional constraint of having to move in the 8 knight Ls.

```python3
class Solution:
    def minKnightMoves(self, x: int, y: int) -> int:
        q = deque([(0, 0)])
        ans = 0
        vis = {(0, 0)}
        dirs = ((-2, 1), (-1, 2), (1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1))
        while q:
            for _ in range(len(q)):
                i, j = q.popleft()
                if (i, j) == (x, y):
                    return ans
                for a, b in dirs:
                    c, d = i + a, j + b
                    if (c, d) not in vis:
                        vis.add((c, d))
                        q.append((c, d))
            ans += 1
        return -1
```

**Time:** O(n^2)
**Space:** O(n^2)

## 108. Cheapest Flights Within K Stops
**Reference:** https://github.com/neetcode-gh/leetcode/blob/main/python/0787-cheapest-flights-within-k-stops.py

**Description:** There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

**Constraints:** 
1 <= n <= 100
0 <= flights.length <= (n * (n - 1) / 2)
flights[i].length == 3
0 <= fromi, toi < n
fromi != toi
1 <= pricei <= 104
There will not be any multiple flights between two cities.
0 <= src, dst, k < n
src != dst

**Examples:** 
```python3
n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1 #=> 700
```

![image](https://github.com/will4skill/algo-review/assets/10373005/4f6965d5-b79b-421f-923e-98e3666ac4e9)


```python3
n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1 #=> 200
```

![image](https://github.com/will4skill/algo-review/assets/10373005/4a52714d-0d39-4c08-a937-b1b3a5a17257)


```python3
n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0 #=> 500
```

![image](https://github.com/will4skill/algo-review/assets/10373005/fed2e1ad-5cc8-4312-8312-a598c41c5b99)

**Hint:** The Neetcode solution is a modified Bellman Ford algorithm, where you consider all edges in every loop and use a temp array to avoid checking beyond the current level.

```python3
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
        prices = [float("inf")] * n
        prices[src] = 0

        for i in range(k + 1):
            tmpPrices = prices.copy()

            for s, d, p in flights:  # s=source, d=dest, p=price
                if prices[s] == float("inf"):
                    continue
                if prices[s] + p < tmpPrices[d]:
                    tmpPrices[d] = prices[s] + p
            prices = tmpPrices
        return -1 if prices[dst] == float("inf") else prices[dst]
```

**Time:** O(E*V)
**Space:** ???

## 109. Word Ladder
**Reference:** https://leetcode.com/problems/word-ladder/solutions/40729/compact-python-solution/

**Description:** A transformation sequence from word beginWord to word endWord using a dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk such that:

1. Every adjacent pair of words differs by a single letter.
2. Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to be in wordList.
3. sk == endWord

Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

**Constraints:** 
1 <= beginWord.length <= 10
endWord.length == beginWord.length
1 <= wordList.length <= 5000
wordList[i].length == beginWord.length
beginWord, endWord, and wordList[i] consist of lowercase English letters.
beginWord != endWord
All the words in wordList are unique.

**Examples:** 
```python3
beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log","cog"] #=> 5
beginWord = "hit", endWord = "cog", wordList = ["hot","dot","dog","lot","log"] #=> 0
```

**Hint:** BFS. Use set to store words. If no endWord in set, return 0. Push beginWord on queue. While queue is not empty, see if you can convert it into another word in set by swapping one character. If so, push it into the queue and remove it from the set. Increment length for each iteration and return length when you reach the endWord.

```python3
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        wordList = set(wordList)
        queue = collections.deque([[beginWord, 1]])
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, length + 1])
        return 0
```

**Time:** O(N*26*(L^2))
**Space:** O(n)

## 110. Longest Increasing Path in a Matrix
**Reference:**  https://leetcode.com/problems/longest-increasing-path-in-a-matrix/discuss/1195189/Javascript-Dynamic-Programming

**Description:** Given an m x n integers matrix, return the length of the longest increasing path in matrix.

From each cell, you can either move in four directions: left, right, up, or down. You may not move diagonally or move outside the boundary (i.e., wrap-around is not allowed).

Given two words, beginWord and endWord, and a dictionary wordList, return the number of words in the shortest transformation sequence from beginWord to endWord, or 0 if no such sequence exists.

**Constraints:** 
m == matrix.length
n == matrix[i].length
1 <= m, n <= 200
0 <= matrix[i][j] <= 2^31 - 1

**Examples:** 
```python3
matrix = [[9,9,4],[6,6,8],[2,1,1]] #=> 4
```

![image](https://github.com/will4skill/algo-review/assets/10373005/e9e6c6f9-9fba-47e1-a31e-55f80de7ba38)

```python3
matrix = [[3,4,5],[3,2,6],[2,2,1]] #=> 4
```

![image](https://github.com/will4skill/algo-review/assets/10373005/7f947e6c-4162-48c7-9301-1a4c7d3f6450)

```python3
matrix = [[1]] #=> 1
```

**Hint:** DFS + memoization. Very similar to structy longest path, except you track prev and check if curr > prev before continuing

```javascript
const longestIncreasingPath = (matrix) => {
  const memo = {};
  let max = 0;
  for(let r = 0; r < matrix.length; r++)
    for(let c = 0; c < matrix[0].length; c++)
      max = Math.max(max, dfs(matrix, r, c, memo, -1 )); // Try each starting point
  return max
};

function dfs(matrix, r, c, memo, prev){
  if(r >= matrix.length || r < 0 || c >= matrix[0].length || c < 0 ||
    matrix[r][c] <= prev)
    return 0;

  const key = r + "," + c;
  if (key in memo) return memo[key];

  const up = dfs(matrix, r - 1, c, memo, matrix[r][c]);
  const down = dfs(matrix, r + 1, c, memo, matrix[r][c]);
  const left = dfs(matrix, r, c - 1, memo, matrix[r][c]);
  const right = dfs(matrix, r, c + 1, memo, matrix[r][c]);

  return memo[key] = 1 + Math.max(up, down, left, right);
}
```

**Time:** O(mn)
**Space:** O(mn)

## 111. Word Search II
**Reference:**  https://leetcode.com/problems/longest-increasing-path-in-a-matrix/discuss/1195189/Javascript-Dynamic-Programming

**Description:** Given an m x n board of characters and a list of strings words, return all words on the board. Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

**Constraints:** 
m == board.length
n == board[i].length
1 <= m, n <= 12
board[i][j] is a lowercase English letter.
1 <= words.length <= 3 * 10^4
1 <= words[i].length <= 10
words[i] consists of lowercase English letters.
All the strings of words are unique.

**Examples:** 
```python3
board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"] #=> ["eat","oath"]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/9ca4b92d-52fd-4275-8772-08b815ecc46a)


```python3
board = [["a","b"],["c","d"]], words = ["abcb"] #=> []
```

![image](https://github.com/will4skill/algo-review/assets/10373005/64078333-9fe1-4436-9a1d-bf4fcd614e95)


**Hint:**
1. Create a Trie data structure
2. Insert all words into trie
3. DFS over mxn board, searching for your words

```python3
class TrieNode:
    def __init__(self):
        self.children = {}
        self.isWord = False
        self.refs = 0

    def addWord(self, word):
        cur = self
        cur.refs += 1
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
            cur.refs += 1
        cur.isWord = True

    def removeWord(self, word):
        cur = self
        cur.refs -= 1
        for c in word:
            if c in cur.children:
                cur = cur.children[c]
                cur.refs -= 1


class Solution:
    def findWords(self, board: List[List[str]], words: List[str]) -> List[str]:
        root = TrieNode()
        for w in words:
            root.addWord(w)

        ROWS, COLS = len(board), len(board[0])
        res, visit = set(), set()

        def dfs(r, c, node, word):
            if (
                r not in range(ROWS) 
                or c not in range(COLS)
                or board[r][c] not in node.children
                or node.children[board[r][c]].refs < 1
                or (r, c) in visit
            ):
                return

            visit.add((r, c))
            node = node.children[board[r][c]]
            word += board[r][c]
            if node.isWord:
                node.isWord = False
                res.add(word)
                root.removeWord(word)

            dfs(r + 1, c, node, word)
            dfs(r - 1, c, node, word)
            dfs(r, c + 1, node, word)
            dfs(r, c - 1, node, word)
            visit.remove((r, c))

        for r in range(ROWS):
            for c in range(COLS):
                dfs(r, c, root, "")

        return list(res)
```

**Time:** O((ROWS * COLS) * (4 * (3 ^ (WORDS - 1))))
**Space:** O(N)

## 112. Alien Dictionary
**Reference:**  https://leetcode.com/problems/alien-dictionary/discuss/763913/Javascript-Graph-Topological-Sort

**Description:** There is a new alien language which uses the latin alphabet. However, the order among letters are unknown to you. You receive a list of non-empty words from the dictionary, where words are sorted lexicographically by the rules of this new language. Derive the order of letters in this language.

**Constraints:** 
1. You may assume all letters are in lowercase.
2. You may assume that if a is a prefix of b, then a must appear before b in the given dictionary.
3. If the order is invalid, return an empty string.
4. There may be multiple valid order of letters, return any one of them is fine.

**Examples:** 
```python3
[
  "wrt",
  "wrf",
  "er",
  "ett",
  "rftt"
]
#=> "wertf"

[
  "z",
  "x"
]
#=> "zx"

[
  "z",
  "x",
  "z"
]
#=> ""
```

**Hint:**
1 Convert the word list into an adjacencyList. To do this, compare each pair of words and use the first differing letter to determine the relative order of edges.
2. Topologically sort your graph

```javascript
const alienOrder = (words) => {
    // Step 0: Create data structures and find all unique letters.
    const adjList = {};
    const numParents = {};
    for (let word of words) {
        for (let c of word) {
            numParents[c] = 0; // intialize parents of each node to zero
            adjList[c] = []; // create empty adjList
        }
    }

    // Step 1: Find all edges.
    for (let i = 0; i < words.length - 1; ++i) {
        const word1 = words[i];
        const word2 = words[i + 1];
        // Check that word2 is not a prefix of word1.
        if (word1.length > word2.length && word1.startsWith(word2)) return "";

        // Find the first non match and insert the corresponding relation.
        for (let j = 0; j < Math.min(word1.length, word2.length); j++) {
            if (word1[j] !== word2[j]) {
                adjList[word1[j]].push(word2[j]);
                numParents[word2[j]]++
                break;
            }
        }
    }

    // Step 2: Breadth-first search.
    // TIM NOTE: this seems like topological sort
    let outputString = "";
    const queue = [];
    for (let c in numParents)
        if (numParents[c] === 0)
          queue.push(c);

    while (queue.length) {
        const curr = queue.shift();
        outputString += curr;
        for (let next of adjList[curr]) {
            numParents[next]--;
            if (numParents[next] === 0)
                queue.push(next);
        }
    }

    if (outputString.length < Object.keys(numParents).length)
        return "";

    return outputString;
}
```

**Time:** O(C), C = total length of all the words
**Space:** O(1) O(U + min U^2, N), N be the total number of strings in the input list, U be the total number of unique letters in the alien alphabet.

## 113. Bus Routes
**Reference:**  https://leetcode.com/problems/bus-routes/solutions/122771/c-java-python-bfs-solution/
ax2

**Description:** You are given an array routes representing bus routes where routes[i] is a bus route that the ith bus repeats forever.

1. For example, if routes[0] = [1, 5, 7], this means that the 0th bus travels in the sequence 1 -> 5 -> 7 -> 1 -> 5 -> 7 -> 1 -> ... forever.
You will start at the bus stop source (You are not on any bus initially), and you want to go to the bus stop target. You can travel between bus stops by buses only.

Return the least number of buses you must take to travel from source to target. Return -1 if it is not possible.

**Constraints:** 
1 <= routes.length <= 500.
1 <= routes[i].length <= 10^5
All the values of routes[i] are unique.
sum(routes[i].length) <= 10^5
0 <= routes[i][j] < 10^6
0 <= source, target < 10^6

**Examples:** 
```python3
routes = [[1,2,7],[3,6,7]], source = 1, target = 6 #=> 2
routes = [[7,12],[4,5,15],[6],[15,19],[9,12,13]], source = 15, target = 12 #=> -1
```

**Hint:** BFS
1 First create an adjacencyList with the routes
2. Use visited to avoid loops
3. Traverse until you reach end

```python3
class Solution:
    def numBusesToDestination(self, routes, S, T):
        stopToRoute = collections.defaultdict(set)
        
        for i, stops in enumerate(routes):
            for stop in stops: 
                stopToRoute[stop].add(i)
                
        bfs = [(S,0)]
        seenStops = {S}
        seenRoutes = set()
        
        for stop, count in bfs:
            if stop == T: 
                return count
            
            for routeIndex in stopToRoute[stop]:
                if routeIndex not in seenRoutes:
                    seenRoutes.add(routeIndex)
                    for next_stop in routes[routeIndex]:
                        if next_stop not in seenStops:
                            seenStops.add(next_stop)
                            bfs.append((next_stop, count+1))
        return -1
```

**Time:** O(m+n) n = edges, m = nodes
**Space:** O(1)

## 114. Lowest Common Ancestor of a Binary Search Tree
**Reference:**  https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/solutions/1347857/c-java-python-iterate-in-bst-picture-explain-time-o-h-space-o-1/

**Description:** Given a binary search tree (BST), find the lowest common ancestor (LCA) node of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

**Constraints:** 
The number of nodes in the tree is in the range [2, 10^5].
-10^9 <= Node.val <= 10^9
All Node.val are unique.
p != q
p and q will exist in the BST.

**Examples:** 
```python3
root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8 #=> 6
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d97889f1-4a56-491c-ad0e-81a23ac48367)

```python3
root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4 #=> 2
```

![image](https://github.com/will4skill/algo-review/assets/10373005/9014c215-2e98-44f3-85c0-3d8f80d739c2)

```python3
root = [2,1], p = 2, q = 1 #=> 2
```

**Hint:** BFS
Note, there is a more general solution for non BSTs. 
1. Let large = max(p.val, q.val), small = min(p.val, q.val)
2. If root.val > large then both node p and q belong to the left subtree, go to left by root = root.left.
3. If root.val < small then both node p and q belong to the right subtree, go to right by root = root.right.
4. Now, small <= root.val <= large the current root is the LCA between q and p.

```python3
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        small = min(p.val, q.val)
        large = max(p.val, q.val)
        while root:
            if root.val > large:  # p, q belong to the left subtree
                root = root.left
            elif root.val < small:  # p, q belong to the right subtree
                root = root.right
            else:  # Now, small <= root.val <= large -> This is the LCA between p and q
                return root
        return None
```

**Time:** O(height)
**Space:** O(1)

## 115. Convert Sorted Array to Binary Search Tree
**Reference:** https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/submissions/1139608527/

**Description:** Given an integer array nums where the elements are sorted in ascending order, convert it to a 
height-balanced binary search tree.

**Constraints:** 
1 <= nums.length <= 10^4
-10^4 <= nums[i] <= 10^4
nums is sorted in a strictly increasing order.

**Examples:** 
```python3
nums = [-10,-3,0,5,9] #=> [0,-3,9,-10,null,5] or [0,-10,5,null,-3,null,9] 
```

![image](https://github.com/will4skill/algo-review/assets/10373005/f1ca1bcb-c778-45b4-9be4-5918ba339111)

or 

![image](https://github.com/will4skill/algo-review/assets/10373005/5e060ac1-9fd4-4150-a5b1-abc61db3c72c)

```python3
nums = [1,3] #=> [3,1] or [1,null,3]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/74904945-b0d6-4f05-b177-3e49331ef022)


**Hint:** Recursively find the middle element (//2) and create a new tree. The left and right branches are recursive calls

```python3
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        if len(nums) == 0:
            return None
        if len(nums) == 1:
            return TreeNode(nums[0])

        start = 0
        end = len(nums) - 1
        mid = start + (end - start) // 2
        left = nums[0:mid]
        right = nums[mid + 1:]

        head = TreeNode(nums[mid])
        head.left = self.sortedArrayToBST(left)
        head.right = self.sortedArrayToBST(right)
        return head
```

**Time:** O(n log n)
**Space:** O(n)

## 116. Validate Binary Search Tree
**Reference:** https://leetcode.com/problems/validate-binary-search-tree/solutions/32178/clean-python-solution/
wz2326

**Description:** Given the root of a binary tree, determine if it is a valid binary search tree (BST).

A valid BST is defined as follows:
1. The left subtree of a node contains only nodes with keys less than the node's key.
2. The right subtree of a node contains only nodes with keys greater than the node's key.
3. Both the left and right subtrees must also be binary search trees.

**Constraints:** 
The number of nodes in the tree is in the range [1, 10^4].
-2^31 <= Node.val <= 2^31 - 1

**Examples:** 
```python3
root = [2,1,3] #=> true
```

![image](https://github.com/will4skill/algo-review/assets/10373005/d7d467a9-25be-4b48-8b9f-a15730232b5f)


```python3
root = [5,1,4,null,null,3,6] #=> false
```

![image](https://github.com/will4skill/algo-review/assets/10373005/33b6bf34-b7cb-4e12-8597-c1330ba3d37a)

**Hint:** See structy. Traverse tree in order, check if values from traversal are sorted. If sorted, return true.

```python3
class Solution(object):
    def isValidBST(self, root, floor=float('-inf'), ceiling=float('inf')):
        if not root: 
            return True
        if root.val <= floor or root.val >= ceiling:
            return False
        # in the left branch, root is the new ceiling; contrarily root is the new floor in right branch
        return self.isValidBST(root.left, floor, root.val) and self.isValidBST(root.right, root.val, ceiling)
```

**Time:** O(n)
**Space:** O(n)

## 117. Kth Smallest Element in a BST
**Reference:** https://leetcode.com/problems/kth-smallest-element-in-a-bst/solutions/1960046/recursion-vector-solutions-with-complexity-analysis-c-java-python/

**Description:** Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

Follow up: If the BST is modified often (i.e., we can do insert and delete operations) and you need to find the kth smallest frequently, how would you optimize?

**Constraints:** 
The number of nodes in the tree is n.
1 <= k <= n <= 10^4
0 <= Node.val <= 10^4


**Examples:** 
```python3
root = [3,1,4,null,2], k = 1 #=> 1
```

![image](https://github.com/will4skill/algo-review/assets/10373005/fe7adb67-2aa3-4b30-a647-741c68c35758)


```python3
root = [5,3,6,2,4,null,null,1], k = 3 #=> 3
```

![image](https://github.com/will4skill/algo-review/assets/10373005/c87fecaf-a65b-4998-a704-0548bc3a1d7c)

**Hint:** Traverse the BST in order and save the results in an array. Return inorder[k-1]

```python3
class Solution(object):
    def kthSmallest(self, root, k):
        values = []
        self.inorder(root, values)
        return values[k - 1]

    def inorder(self, root, values):
        if root is None:
            return
        self.inorder(root.left, values)
        values.append(root.val)
        self.inorder(root.right, values)
```

**Time:** O(n)
**Space:** O(n)

## 118. Inorder Successor in BST
**Reference:** https://www.enjoyalgorithms.com/blog/inorder-successor-in-binary-search-tree

**Description:** Given a binary search tree and a node in it, find the in-order successor of that node in the BST.
The successor of a node p is the node with the smallest key greater than p.val.

**Constraints:** 
1. If the given node has no in-order successor in the tree, return null.
2. It's guaranteed that the values of the tree are unique.


**Examples:** 
```python3
root = [2,1,3], p = 1 #=> 2
```

![image](https://github.com/will4skill/algo-review/assets/10373005/c8ab780d-b161-4d8a-95e5-d34d2bc983b0)


```python3
root = [5,3,6,2,4,null,null,1], p = 6 #=> null
```

![image](https://github.com/will4skill/algo-review/assets/10373005/60d2d944-f9e9-447b-97d9-a9444c796026)


**Hint:** Create a method to recurively find minimum value in BST. 

1. Case 1: if the right child of the target node is present, then the success is the minimum value in the right subtree
2. Case 2. if the right child is not present, traverse upward to find the successor
   a. if root.val > target.val, let succPtr = root, recurse on left subtree (root.left)
   b. if root.val < target.val, recurse on right subtree (root.right)

```python3
class Solution(object):
    def bst_minimum(root):
        while root.left is not None:
            root = root.left
        return root

    def inorder_successor(root, succ, x):
        if root is None:
            return succ

        if x.key == root.key:
            if x.right is not None:
                return bst_minimum(x.right)
        elif x.key < root.key:
            succ = root
            return inorder_successor(root.left, succ, x)
        else:
            return inorder_successor(root.right, succ, x)
        return succ
```

**Time:** O(height)
**Space:** O(height)

## 119. Ransom Note
**Reference:** https://leetcode.com/problems/ransom-note/submissions/1135825421/

**Description:** Given two strings ransomNote and magazine, return true if ransomNote can be constructed by using the letters from magazine and false otherwise. Each letter in magazine can only be used once in ransomNote.

**Constraints:** 
1. 1 <= ransomNote.length, magazine.length <= 10^5
2. ransomNote and magazine consist of lowercase English letters.

**Examples:** 
```python3
ransomNote = "a", magazine = "b" #=> false
ransomNote = "aa", magazine = "ab" #=> false
ransomNote = "aa", magazine = "aab" #=> true
```

**Hint:** Create a hashMap counting the letters in magazine. Iterate over ransomeNote subtracting letters from the hm. If you run out, return false. 

```python3
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        if len(ransomNote) > len(magazine): return False
        charMap = {}

        for char in magazine:
            if charMap.get(char):
                charMap[char] += 1
            else: 
                charMap[char] = 1
        
        for char in ransomNote:
            if charMap.get(char, 0) > 0:
                charMap[char] -= 1
            else:
                return False
        
        return True
```

**Time:** O(n)
**Space:** O(n)

## 120. Insert Delete GetRandom O(1)
**Reference:** https://leetcode.com/problems/insert-delete-getrandom-o1/solutions/455253/python-super-efficient-detailed-explanation/

**Description:** Implement the RandomizedSet class:
1. RandomizedSet() Initializes the RandomizedSet object.
2. bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
3. bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
4. int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.
You must implement the functions of the class such that each function works in average O(1) time complexity.

**Constraints:** 
1. -2^31 <= val <= 2^31 - 1
2. At most 2 * 10^5 calls will be made to insert, remove, and getRandom.
3. There will be at least one element in the data structure when getRandom is called.

**Examples:** 
```python3
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []] #=> [null, true, false, true, 2, true, false, 2]
```

**Hint:** If it wasn't for get random, you could just use a set. Instead use a list that holds the values and a map between each value and its location in the list. That way you can use the list when you want a random element.

Insert: just append to the end of the list and add a new entry to the hm
delete: Swap the last element in the list with the one you are removing, pop it from the list O(1) and remove it from the hm O(1)

```python3
class RandomizedSet:
    def __init__(self):
        self.data_map = {} # dictionary, aka map, aka hashtable, aka hashmap
        self.data = [] # list aka array
    def insert(self, val: int) -> bool:
        # the problem indicates we need to return False if the item 
        # is already in the RandomizedSet---checking if it's in the
        # dictionary is on average O(1) where as
        # checking the array is on average O(n)
        if val in self.data_map:
            return False

        # add the element to the dictionary. Setting the value as the 
        # length of the list will accurately point to the index of the 
        # new element. (len(some_list) is equal to the index of the last item +1)
        self.data_map[val] = len(self.data)

        # add to the list
        self.data.append(val)
        
        return True

    def remove(self, val: int) -> bool:
        # again, if the item is not in the data_map, return False. 
        # we check the dictionary instead of the list due to lookup complexity
        if not val in self.data_map:
            return False

        # essentially, we're going to move the last element in the list 
        # into the location of the element we want to remove. 
        # this is a significantly more efficient operation than the obvious 
        # solution of removing the item and shifting the values of every item 
        # in the dicitionary to match their new position in the list
        last_elem_in_list = self.data[-1]
        index_of_elem_to_remove = self.data_map[val]

        self.data_map[last_elem_in_list] = index_of_elem_to_remove
        self.data[index_of_elem_to_remove] = last_elem_in_list

        # change the last element in the list to now be the value of the element 
        # we want to remove
        self.data[-1] = val

        # remove the last element in the list
        self.data.pop()

        # remove the element to be removed from the dictionary
        self.data_map.pop(val)
        return True

    def getRandom(self) -> int:
        # if running outside of leetcode, you need to `import random`.
        # random.choice will randomly select an element from the list of data.
        return random.choice(self.data)
```

**Time:** O(1)
**Space:** O(n)

## 121. First Missing Positive
**Reference:** https://leetcode.com/problems/first-missing-positive/solutions/17080/python-o-1-space-o-n-time-solution-with-explanation/
@ivalue

**Description:** Given an unsorted integer array nums, return the smallest missing positive integer. You must implement an algorithm that runs in O(n) time and uses O(1) auxiliary space.

**Constraints:** 
1. 1 <= nums.length <= 10^5
2. -2^31 <= nums[i] <= 2^31 - 1

**Examples:** 
```python3
nums = [1,2,0] #=> 3
nums = [3,4,-1,1] #=> 2
nums = [7,8,9,11,12] #=> 1
```

**Hint:** 
https://leetcode.com/problems/first-missing-positive/solutions/17080/python-o-1-space-o-n-time-solution-with-explanation/   
 1. for any array whose length is l, the first missing positive must be in range [1,...,l+1], 
        so we only have to care about those elements in this range and remove the rest.
2. we can use the array index as the hash to restore the frequency of each number within 
         the range [1,...,l+1] 

```python3
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        nums = list(set(nums)) + [0]
        n = len(nums)
        for i in range(len(nums)):  # delete those useless elements
            if nums[i] < 0 or nums[i] >= n:
                nums[i] = 0
        for i in range(len(nums)):  # use the index as the hash to record the frequency of each number
            nums[nums[i] % n] += n
        for i in range(1, len(nums)):
            if nums[i] // n == 0:
                return i
        return n
```

**Time:** O(n)
**Space:** O(1)

## 122. Climbing Stairs
**Reference:** https://leetcode.com/problems/climbing-stairs/solutions/3708750/4-method-s-beat-s-100-c-java-python-beginner-friendly/

**Description:** You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

**Constraints:** 
1 <= n <= 45

**Examples:** 
```python3
n = 2 #=> 2
n = 3 #=> 3
```

**Hint:** Dynamic Programming.
Just fibonacci. You can memoize, because you don't need specific paths, just the counts. Take either one step or two until steps remaing are 0 or negative

```python3
class Solution:
    def climbStairs(self, n: int) -> int:
        memo = {}
        return self.helper(n, memo)
    
    def helper(self, n: int, memo: dict[int, int]) -> int:
        if n == 0 or n == 1:
            return 1
        if n not in memo:
            memo[n] = self.helper(n-1, memo) + self.helper(n-2, memo)
        return memo[n]
```

**Time:** O(n)
**Space:** O(n) Note: O(1) space possible, if you just use two previous values

## 123. Maximum Subarray
**Reference:** https://leetcode.com/problems/maximum-subarray

**Description:** Given an integer array nums, find the subarray with the largest sum, and return its sum.

Follow up: If you have figured out the O(n) solution, try coding another solution using the divide and conquer approach, which is more subtle.

**Constraints:** 
1 <= nums.length <= 10^5
-10^4 <= nums[i] <= 10^4

**Examples:** 
```python3
nums = [-2,1,-3,4,-1,2,1,-5,4] #=> 6 *[4,-1,2,1]*
nums = [1] #=> 1
nums = [5,4,-1,7,8] #=> 23 *[5,4,-1,7,8]*
```

**Hint:** Dynamic Programming.
Extend (in one direction) or start again. At each element in the array you have two choices: append current sum with new element or start over with new element. Pick whichever gives the highest value. Since your local max might be wiped out by negative numbers, also maintain a global max so that you can return the global max value at the end.

```javascript
const maxSubArray = nums => {
  let localMaxSum = nums[0];
  let globalMaxSum = nums[0];

  for (let i = 1; i < nums.length; ++i) {
      let num = nums[i];
    localMaxSum = Math.max(num, localMaxSum + num);
    globalMaxSum = Math.max(globalMaxSum, localMaxSum);
  }
  return globalMaxSum;
}
```

**Time:** O(n)
**Space:** O(1)

## 124. Coin Change
**Reference:** https://www.structy.net/problems/min-change

**Description:** You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

**Constraints:** 
1 <= coins.length <= 12
1 <= coins[i] <= 2^31 - 1
0 <= amount <= 10^4

**Examples:** 
```python3
coins = [1,2,5], amount = 11 #=> 3
coins = [2], amount = 3 #=> -1
coins = [1], amount = 0 #=> 0
```

**Hint:** Dynamic Programming.
DFS + memoization:
MemoKey: target => # of coins

Base cases: 
1. if target is negative => Infinity (don't follow that path)
2. if target == 0 => return 0 (success)
3. if target in memo => return memo[target]

Branches: 
1. Each branch is a different coin being used once
2. When you recurse remember to subtract that coin from the target
3. Track the minimum recursive branch result

Final return: 1 for current coin + memo[target]

```python3
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        ans = self.helper(amount, coins, {})
        if ans == float('inf'): return -1
        else: return ans
    
    def helper(self, amount, coins, memo):
        if amount in memo: return memo[amount]
        if amount == 0: return 0
        if amount < 0: return float('inf')
        
        min_coins = float('inf')
        for coin in coins:
            min_coins = min(min_coins, self.helper(amount - coin, coins, memo))
            
        memo[amount] = min_coins + 1
        return memo[amount]
```

**Time:** O(a*c) a = amount c = # coins
**Space:** O(a)

## 125. Partition Equal Subset Sum
**Reference:** https://techsauce.medium.com/solving-partition-equal-subset-sum-problem-knapsack-problem-2b47ad13733b

**Description:** Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

**Constraints:** 
1 <= nums.length <= 200
1 <= nums[i] <= 100

**Examples:** 
```python3
nums = [1,5,11,5] #=> true
nums = [1,2,3,5] #=> false
```

**Hint:** Dynamic Programming.
Knapsack type problem. You can use DFS and memoization

first, target = sum(nums) / 2
if sum(nums) is odd return false

Memo key: (target,idx)

Base cases:
1. if target == 0: return True
2. if target is negtive or index is past nums.length - 1 return False
3. If there is already a value for the target at the current idx pull it from the memo 

Branches: 
1. Include curr num (target - nums[idx]), increment idx 
2. Skip curr num (target is unchanged), increment idx 

Final Return: memoize (target, idx) = result
return result

```python3
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        # Calculate the total sum of all numbers in the array
        total = 0
        for i in nums:
            total += i
        # If the total sum is odd, it's not possible to partition the array into two subsets of equal sums
        if total % 2 != 0:
            return False
        # Start the recursion from the first number (index 0) and current sum 0
        return self.canPartitionFrom(nums, 0, 0, total // 2, memo = {})

    def canPartitionFrom(self, nums, index, sum, target, memo):
        key = (index, sum)
        # If the current sum equals the target sum, we've found a valid subset
        if sum == target:
            return True
        # If the current sum exceeds the target, or we've tried all numbers, stop the recursion
        if sum > target or index >= len(nums):
            return False
        # If we've already computed the result for this state (index, sum), return it from memo
        if key in memo:
            return memo[key]
        # Otherwise, compute the result:
        # Try to include the current number in the subset (and add it to the sum) OR
        # try to exclude the current number (and leave the sum as it is)
        result = (self.canPartitionFrom(nums, index + 1, sum, target, memo) or 
        self.canPartitionFrom(nums, index + 1, sum + nums[index], target, memo))

        # Store the result in the dp table for future reference
        memo[key] = result

        return result
```

**Time:** O(N*sum)
**Space:** O(N*sum)

## 126. Unique Paths
**Reference:** https://leetcode.com/problems/unique-paths/solutions/1670399/js-dp-memo-and-tabulation/

**Description:** There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 10^9.

**Constraints:** 
1 <= m, n <= 100

**Examples:** 
```python3
m = 3, n = 7 #=> 28
```

![image](https://github.com/will4skill/algo-review/assets/10373005/97906486-a369-4d98-ae3a-fc9de2c8021a)

```python3
m = 3, n = 2 #=> 3
```

**Hint:** Dynamic Programming.
You can use DFS and memoization

Memo key: (m,n)

Base cases:
1. if key in memo return memo[key]
2. if m == 1 && n == 1 return 1 (success)
3. if m == 0 || n == 0 return 0 (failure) 

Branches: 
1. go down (m -= 1)
2. go right (n -= 1)

Return: memoize sum of both branches
return memo

```python3
class Solution:
    def uniquePaths(self, m: int, n: int, memo = {}) -> int:
        key = (m, n)
        if key in memo: return memo[key]
        if m == 1 and n == 1: return 1
        if m == 0 or n == 0: return 0
        memo[key] = self.uniquePaths(m, n - 1, memo) + self.uniquePaths(m - 1, n, memo)
        return memo[key]
```

**Time:** O(nm)
**Space:** O(nm)

## 127. House Robber
**Reference:** https://leetcode.com/problems/unique-paths/solutions/1670399/js-dp-memo-and-tabulation/

**Description:** You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security systems connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

**Constraints:** 
1 <= nums.length <= 100
0 <= nums[i] <= 400

**Examples:** 
```python3
nums = [1,2,3,1] #=> 4
nums = [2,7,9,3,1] #=> 12
```

**Hint:** Dynamic Programming.
You can use DFS and memoization

Memo key: idx // because order doesn't matter

Base cases:
1. If key in memo return memo[key]
2. If idx >= nums.length return 0 // Fail

Branches: 
1. currHouse + dfs(idx + 2), dfs(idx + 1)

Return: memoise Max of braches above
return memo

```python3
class Solution:
    def rob(self, nums: List[int]) -> int:
        return self.helper(nums, len(nums) - 1, {})
    def helper(self, nums, idx, memo):
        if idx < 0: return 0
        if idx in memo: return memo[idx]
        memo[idx] = max(self.helper(nums, idx - 2, memo) + nums[idx], self.helper(nums, idx - 1, memo))
        return memo[idx]
```

**Time:** O(n)
**Space:** O(n)

## 128. Maximum Product Subarray (*contiguous)
**Reference:** https://leetcode.com/problems/maximum-product-subarray/solutions/48302/2-passes-scan-beats-99/

**Description:** Given an integer array nums, find a subarray that has the largest product, and return the product. The test cases are generated so that the answer will fit in a 32-bit integer.

**Constraints:** 
1 <= nums.length <= 2 * 10^4
-10 <= nums[i] <= 10
The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

**Examples:** 
```python3
nums = [2,3,-2,4] #=> 6
nums = [-2,0,-1] #=> 0
```

**Hint:** Dynamic Programming.
Similar to max sum subarray, you have a global max and a local max. At each item in the array, the new local max will either be the new item by itself or the new item * the currentLocal max. If a zero is encountered, After updating the localMax, update the globalMax if possible. reset the localMax to 0. To account for negative numbers, you have to repeat this process from right to left as well. 

```python3
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        maxNum, product, length = float('-inf'), 1, len(nums)
        for idx in range(length):
            product *= nums[idx]
            maxNum = max(product, maxNum)
            if nums[idx] == 0: product = 1
        product = 1
        for idx in range(length-1, -1, -1):
            product *= nums[idx]
            maxNum = max(product, maxNum)
            if nums[idx] == 0: product = 1
        
        return maxNum
```

**Time:** O(n)
**Space:** O(n)

## 129. Longest Increasing Subsequence (*!= contiguous)
**Reference:** https://leetcode.com/problems/longest-increasing-subsequence/solutions/1326552/optimization-from-brute-force-to-dynamic-programming-explained/

Follow up: Can you come up with an algorithm that runs in O(n log(n)) time complexity?

**Description:** Given an integer array nums, return the length of the longest strictly increasing subsequence.

**Constraints:** 
1 <= nums.length <= 2500
-10^4 <= nums[i] <= 10^4

**Examples:** 
```python3
nums = [10,9,2,5,3,7,101,18] #=> 4
nums = [0,1,0,3,2,3] #=> 4
nums = [7,7,7,7,7,7,7] #=> 1
```

**Hint:** Dynamic Programming.
You can use DFS and memoization

Memo key: idx,previousNumber

Base cases:
1. If key in memo return memo[key]
2. If idx >= nums.length return 0 // Fail

Branches: 
1. skip the current number recurse with idx + 1, and pass in previous
2. include the current number (add 1 to result) recurse with idx + 1 and pass in current

Return: memoize Max of braches above
return memo

```python3
# Correct But TLE... 
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        return self.helper(nums, 0, -1, {})

    def helper(self, nums, idx, prevIdx, memo):
        key = (idx, prevIdx + 1)
        if idx >= len(nums): return 0
        if key in memo: return memo[key]

        take = 0
        dontTake = self.helper(nums, idx + 1, prevIdx, memo)
        if prevIdx == -1 or nums[idx] > nums[prevIdx]:
            # try picking current element if no previous element 
            # is chosen or current > nums[prevIdx]
            take = 1 + self.helper(nums, idx + 1, idx, memo)
        memo[key] = max(take, dontTake)
        return memo[key]
```

**Time:** O(n^2)
**Space:** O(N)

```python3
# Bottom up is faster
class Solution:
    def lengthOfLIS(self, nums):
        ans = 1
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
                    ans = max(ans, dp[i])
        return ans
```

**Time:** O(n^2)
**Space:** O(N)

```python3
class Solution:
    def lengthOfLIS(self, A):
        len = 0
        for cur in A:
            if len == 0 or A[len-1] < cur:
                A[len] = cur  # extend
                len += 1
            else:
                # replace
                from bisect import bisect_left
                A[bisect_left(A, cur, 0, len)] = cur
        return len
```

**Time:** O(nlog n)
**Space:** O(1)

## 130. Jump Game
**Reference:** https://leetcode.com/problems/jump-game/solutions/2375320/interview-scenario-recursion-memoization-dp-greedy/

**Description:** You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

**Constraints:** 
1 <= nums.length <= 10^4
0 <= nums[i] <= 10^5

**Examples:** 
```python3
nums = [2,3,1,1,4] #=> true
nums = [3,2,1,0,4] #=> false
```

**Hint:** Dynamic Programming.
Simplest: iterate over array once, updating the max distance you can travel until you care capable of reaching the end or finish iterating without being capable.

DFS with memo (O(n^2) time, O(n) + O(n) stack space and memo space)

Memo key: idx 

Base cases:
1. If key in memo return memo[key]
2. If idx == nums.length - 1 return true
3. if nums[idx] == 0 return false

Branches: 
1. try every jump size from 1 to maximum (nums[idx]). If dfs returns true memoize it and return memo

Return: if you reach the outside of the for loop, memoize result
return memo

```python3
# Correct But TLE... 
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        return self.helper(nums, 0, {})

    def helper(self, nums, idx, memo):
        if idx == len(nums) - 1: return True
        if nums[idx] == 0: return False
        if idx in memo: return memo[idx]

        reach = idx + nums[idx]
        for jump in range(idx + 1, reach + 1):
            if jump < len(nums) and self.helper(nums, jump, memo):
                memo[idx] = True  # memoizing for a particular index
                return memo[idx]

        memo[idx] = False
        return memo[idx]
```

**Time:** O(N* N) -> for each index, I can have at max N jumps, hence O(N* N).
**Space:** O(N) + O(N) -> stack space plus dp array size.

```python3
# Bottom Up 
class Solution:
    def canJump(self, nums):
        n = len(nums)
        dp = [-1] * n
        dp[n - 1] = 1  # base case

        for idx in range(n - 2, -1, -1):
            if nums[idx] == 0:
                dp[idx] = False
                continue

            flag = 0
            reach = idx + nums[idx]
            for jump in range(idx + 1, reach + 1):
                if jump < n and dp[jump]:
                    dp[idx] = True
                    flag = 1
                    break
            if flag == 1:
                continue

            dp[idx] = False

        return dp[0]
```

**Time:** O(N* N)
**Space:** O(N) -> dp array size

```python3
# Kadane
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        reach = 0
        for idx in range(len(nums)):
            if reach < idx:
                return False
            reach = max(reach, idx + nums[idx])
        return True
```

**Time:** O(n)
**Space:** O(1)

## 131. Maximal Square
**Reference:** https://leetcode.com/problems/maximal-square/solutions/473270/all-four-approaches-in-c-brute-force-recursive-dp-memoization-tabulation/

**Description:** Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

**Constraints:** 
m == matrix.length
n == matrix[i].length
1 <= m, n <= 300
matrix[i][j] is '0' or '1'.

**Examples:** 
```python3
matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]] #=> 4
```

![image](https://github.com/will4skill/algo-review/assets/10373005/f132a81a-7bad-4f33-afe4-2ca82208b1e7)

```python3
matrix = [["0","1"],["1","0"]] #=> 1
```

![image](https://github.com/will4skill/algo-review/assets/10373005/f132a81a-7bad-4f33-afe4-2ca82208b1e7)

```python3
matrix = [["0"]] #=> 0
```

![image](https://github.com/will4skill/algo-review/assets/10373005/be88b138-e5da-4ee2-ad95-aa46b76e2017)


**Hint:** Dynamic Programming.
Memo Version:
Memo key: (i,j)

Base cases:
1. If key in memo return memo[key]
2. If i or j out of bounds, return 0
3. if matrix[i][j] = 0

Branches: 
1. try to brach to lower, right, and lower right cells record the results.

Return: store smallest branch in memo
return memo

Bottom Up Version: https://leetcode.com/problems/maximal-square/solutions/600149/python-thinking-process-diagrams-dp-approach/
You can either create a grid the same size as the matrix or use the matrix itself. Iterate from the top left down to the bottom right. 

For each cell check the surrounding three cells and update current (matrix[i][j] to the minimum neighbor + 1). Update the global max if current cell is greater than the global max value.

Return the global max.

"The key is noticing that a square of size 3 is made up of 4 overlapping squares of size 2. This holds as the square size increases, and is why the min(1, 1, 1) + 1 part of the equation works and helps you build up the memo table."

```python3
# Memo: https://leetcode.com/problems/maximal-square/solutions/473270/all-four-approaches-in-c-brute-force-recursive-dp-memoization-tabulation/
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        memo = {}
        res = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                res = max(res, self.solve(matrix, i, j, memo))
        return res * res

    def solve(self, matrix, i, j, memo):
        key = (i,j)
        if i >= len(matrix) or j >= len(matrix[0]): return 0
        if key in memo:
            return memo[key]
        if matrix[i][j] == '0':
            memo[key] = 0
            return 0
        
        memo[key] = min(
            self.solve(matrix, i + 1, j, memo), 
            self.solve(matrix, i, j + 1, memo),
            self.solve(matrix, i + 1, j + 1, memo)
        ) + 1
        return memo[key]
```

**Time:** O(mn)
**Space:** O(mn)??

```python3
# https://leetcode.com/problems/maximal-square/solutions/817156/python-easy-going-from-brute-force-to-dp-solution/
class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        if not matrix:
            return 0
        maxRows = len(matrix)
        maxCols = len(matrix[0])
        maxLength = 0
        dp = [[0 for _ in range(maxCols + 1)] for _ in range(maxRows + 1)]
        for row in range(1, maxRows + 1):
            for col in range(1, maxCols + 1):
                if matrix[row-1][col-1] == '1':
                    dp[row][col] = min(dp[row-1][col-1], dp[row-1][col], dp[row][col-1]) + 1
                    maxLength = max(maxLength, dp[row][col])
        return maxLength**2
```

**Time:** O(mn)
**Space:** O(1)

## 132. Decode Ways
**Reference:** https://leetcode.com/problems/decode-ways/solutions/4454173/recursive-top-down-bottom-up-clean-and-commented-code-dynamic-programming/

**Description:** A message containing letters from A-Z can be encoded into numbers using the following mapping:

'A' -> "1"
'B' -> "2"
...
'Z' -> "26"
To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

"AAJF" with the grouping (1 1 10 6)
"KJF" with the grouping (11 10 6)
Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

Given a string s containing only digits, return the number of ways to decode it.

The test cases are generated so that the answer fits in a 32-bit integer.

**Constraints:** 
1 <= s.length <= 100
s contains only digits and may contain leading zero(s).

**Examples:** 
```python3
s = "12" #=> 4
s = "226" #=> 3
s = "06" #=> 0
```

**Hint:** Dynamic Programming.
*Key: you need to know how to slice strings and parse

Memo key: idx 

Base cases:
1. If key in memo return memo[key]
2. s[i] == "0" return 0
3. if idx == s.length return 1 // Out of bounds

Branches: 
1. Decode a single digit (always possible) and dfs(idx + 1)
2. If idx is valid and the curr and next char make a valid number decode 2 digits and dfs(idx + 2)

Return: memoize curr idx => sum of way1 and way2
return memo

```python3
class Solution:
    def numDecodings(self, s: str) -> int:
        return self.helper(s, 0, {})
        
    def helper(self, s, idx, memo):
        if idx in memo: return memo[idx]
        if idx == len(s): return 1
        if s[idx] == '0': return 0

        ways = self.helper(s, idx + 1, memo)
        if idx + 1 < len(s) and int(s[idx:idx+2]) <= 26:
            ways += self.helper(s, idx + 2, memo)
        
        memo[idx] = ways
        return memo[idx]
```

**Time:** O(n)
**Space:** O(n) due to the memoization table.

```python3
class Solution(object):
    def numDecodings(self, s):
        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1  # Base case: empty string is one valid decoding

        for i in range(n - 1, -1, -1):
            if s[i] == '0':
                dp[i] = 0
            else:
                dp[i] = dp[i + 1]
                if i + 1 < n and int(s[i:i+2]) <= 26:
                    dp[i] += dp[i + 2]

        return dp[0]
```

**Time:** O(n)
**Space:** O(n) due to the 1D array.

## 133. Combination Sum IV
**Reference:** https://leetcode.com/problems/combination-sum-iv/solutions/1166177/short-easy-w-explanation-optimization-from-brute-force-to-dp-solution/

**Description:** Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target. The test cases are generated so that the answer can fit in a 32-bit integer.

Follow up: What if negative numbers are allowed in the given array? How does it change the problem? What limitation we need to add to the question to allow negative numbers?

**Constraints:** 
1 <= nums.length <= 200
1 <= nums[i] <= 1000
All the elements of nums are unique.
1 <= target <= 1000

**Examples:** 
```python3
nums = [1,2,3], target = 4 #=> 7
nums = [9], target = 3 #=> 0
```

**Hint:** Dynamic Programming.
Memo key: target
Memo value: number of combinations 

Base cases:
1. If key in memo return memo[key]
2. target == 0 return 1
3. target < 0 return 0

Branches: 
1. for each num in nums dfs(target - num)

Return: memoize curr target => sum of all ways
return memo

```python3
class Solution:
    def combinationSum4(self, nums, target):
        return self.helper(nums, target, {})

    def helper(self, nums, target, memo):
        if target == 0: return 1  # base condition
        if target in memo: return memo[target]  # if already computed for this value

        count = 0
        # check for every element of nums. An element can only be taken if it is less than target.
        # If an element is chosen, recurse for the remaining value.
        for num in nums:
            if num <= target:
                count += self.helper(nums, target - num, memo)
        memo[target] = count
        return memo[target]
```

**Time:** O(N * T), N = number of elements in nums, T = target value
**Space:** O(T)

```python3
class Solution:
    def combinationSum4(self, nums, target):
        dp = [0] * (target + 1)
        dp[0] = 1
        for curTarget in range(1, target + 1):
            for num in nums:
                if num <= curTarget:
                    dp[curTarget] += dp[curTarget - num]
        return dp[target]
```

**Time:** O(N * T)
**Space:** O(T)

## 134. Add Binary
**Reference:** https://leetcode.com/problems/add-binary/submissions/1136649211/

**Description:** Given two binary strings a and b, return their sum as a binary string.

**Constraints:** 
1 <= a.length, b.length <= 10^4
a and b consist only of '0' or '1' characters.
Each string does not contain leading zeros except for the zero itself.

**Examples:** 
```python3
a = "11", b = "1" #=> "100"
a = "1010", b = "1011" #=> "10101"
```

**Hint:** Start at far right and iterate backward and continue will either string is >= 0 or carry bit is 1. If one string runs out of characters early, don't add it, or make its current char 0

```python3
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        aIdx = len(a) - 1
        bIdx = len(b) - 1
        carry = 0
        output = ""

        while aIdx >= 0 or  bIdx >= 0 or carry == 1:
            if aIdx < 0:
                aChar = 0
            else:
                aChar = a[aIdx]

            if bIdx < 0:
                bChar = 0
            else:
                bChar = b[bIdx]

            combo = int(aChar) + int(bChar) + carry
            if combo == 3:
                output = "1" + output
                carry = 1
            elif combo == 2:
                output = "0" + output
                carry = 1
            elif combo == 1:
                output = "1" + output
                carry = 0
            elif combo == 0:
                output = "0" + output
                carry = 0

            aIdx -= 1
            bIdx -= 1
    

        return output
```

**Time:** O(max(M, N)
**Space:** O(max(M, N)

## 135. Counting Bits
**Reference:** https://leetcode.com/problems/add-binary/submissions/1136649211/

**Description:** Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

It is very easy to come up with a solution with a runtime of O(n log n). Can you do it in linear time O(n) and possibly in a single pass?
Can you do it without using any built-in function (i.e., like __builtin_popcount in C++)?

**Constraints:** 
0 <= n <= 10^5

**Examples:** 
```python3
n = 2 #=> [0,1,1]
n = 5 #=> [0,1,1,2,1,2]
```

**Hint:** Dynamic Programming with Bit Manipulation

shifting a number to the right by one bit (i.e., dividing by 2) removes the last bit. So, the number of 1's in the binary representation of i is the same as i/2 i plus the last bit of i

We use bitwise shift and AND operations. Bitwise right shifting i >> 1 essentially removes the last bit, and i & 1 extracts the last bit. This helps us compute the result for i using previously computed results.

Initialization: Create an array ans of length n + 1, initialized with zeros.
Main Algorithm: Iterate from 1 to n, and for each i, set ans[i] = ans[i >> 1] + (i & 1).

```python3
class Solution:
    def countBits(self, n: int) -> List[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)
        return ans
```

**Time:** O(n)
**Space:** O(n)

## 136. Number of 1 Bits
**Reference:** https://leetcode.com/problems/counting-bits/solutions/3986178/97-97-dp-bit-manipulation-offset/

**Description:** Write a function that takes the binary representation of an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:
1. Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
2. In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.

Follow up: If this function is called many times, how would you optimize it?

**Constraints:** 
The input must be a binary string of length 32.

**Examples:** 
```python3
n = 00000000000000000000000000001011 #=> 3
n = 00000000000000000000000010000000 #=> 1
n = 11111111111111111111111111111101 #=> 31
```

**Hint:** Do this: n=n&(n-1). Change the first set bit from right to 0 until n == 0

```python3
class Solution:
    def hammingWeight(self, n: int) -> int:
        cnt = 0  # count of set bit
        while n > 0:  # iterate until all bits are traversed
            cnt += 1
            n = n & (n - 1)  # change the first set bit from right to 0
        return cnt
```

**Time:** O(logn)
**Space:** O(1)

## 137. Single Number
**Reference:** https://leetcode.com/problems/counting-bits/solutions/3986178/97-97-dp-bit-manipulation-offset/

**Description:** Given a non-empty array of integers nums, every element appears twice except for one. Find that single one. You must implement a solution with a linear runtime complexity and use only constant extra space.

**Constraints:** 
1 <= nums.length <= 3 * 10^4
-3 * 10^4 <= nums[i] <= 3 * 10^4
Each element in the array appears twice except for one element which appears only once.

**Examples:** 
```python3
nums = [2,2,1] #=> 1
nums = [4,1,2,1,2] #=> 4
nums = [1] #=> 1
```

**Hint:** If you XOR a number with itself, 0 is returned. 

Try to xor every value in array with itself return the result

xor ^= num

```python3
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        xor = 0
        for num in nums:
            xor ^= num
        return xor
```

**Time:** O(n)
**Space:** O(1)

## 138. Missing Number
**Reference:** https://leetcode.com/problems/missing-number/solutions/69791/4-line-simple-java-bit-manipulate-solution-with-explaination/ 
shank1499

**Description:** Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?

**Constraints:** 
n == nums.length
1 <= n <= 10^4
0 <= nums[i] <= n
All the numbers of nums are unique.

**Examples:** 
```python3
nums = [3,0,1] #=> 2
nums = [0,1] #=> 2
nums = [9,6,4,2,3,5,7,0,1] #=> 8
```

**Hint:** Two xor operations with the same number will eliminate the number and reveal the original number.

a^b^b = a

Apply XOR operation to both the index and value of the array. In a complete array with no missing numbers, the index and value should be perfectly corresponding(nums[index] = index), so in a missing array, what left finally is the missing number.

```python3
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        result = len(nums)
        for idx in range(len(nums)):
            result = result ^ idx ^ nums[idx] # a^b^b = a
        return result
```

**Time:** O(n)
**Space:** O(1)

## 139. Reverse Bits
**Reference:** https://leetcode.com/problems/reverse-bits/solutions/3218837/190-solution-step-by-step-explanation/

**Description:** Reverse bits of a given 32 bits unsigned integer.
Note:
1. Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
2. In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.

Follow up: If this function is called many times, how would you optimize it?

**Constraints:** 
The input must be a binary string of length 32

**Examples:** 
```python3
n = 00000010100101000001111010011100 #=> 964176192 (00111001011110000010100101000000)
n = 11111111111111111111111111111101 #=> 3221225471 (10111111111111111111111111111111)
```

**Hint:** 
// Iterate over all 32 bits of the given number
for i in range(32):
   // Left shift the reversed number by 1 and add the last bit of the given number to it
   reversed_num = (reversed_num << 1) | (n & 1)
   // To add the last bit of the given number to the reversed number, perform an AND operation with the given number and 1
   n >>= 1 # divid by 2


```python3
class Solution:
    def reverseBits(self, n: int) -> int:
        reversed_num = 0 # Initialize the reversed number to 0
        for i in range(32): # Iterate over all 32 bits of the given number
            # Left shift the reversed number by 1 and add the last bit of the given number to it
            reversed_num = (reversed_num << 1) | (n & 1)
            # To add the last bit of the given number to the reversed number, perform an AND operation with the given number and 1
            n >>= 1 # divid by 2
        return reversed_num # Return the reversed number
```

**Time:** O(1)
**Space:** O(1)

## 140. Find the Duplicate Number
**Reference:** https://leetcode.com/problems/find-the-duplicate-number/solutions/1893098/bit-manipulation-explained/

**Description:** Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

There is only one repeated number in nums, return this repeated number.

You must solve the problem without modifying the array nums and uses only constant extra space.

How can we prove that at least one duplicate number must exist in nums?

Can you solve the problem in linear runtime complexity?

**Constraints:** 
1. 1 <= n <= 10^5
2. nums.length == n + 1
3. 1 <= nums[i] <= n
4. All the integers in nums appear only once except for precisely one integer which appears two or more times.

**Examples:** 
```python3
nums = [1,3,4,2,2] #=> 2
nums = [3,1,3,4,2] #=> 3
```

**Hint:** 
For each possible bit used in the numbers in [1,n] we will count how many of the numbers in nums use that bit, and compare that with the count we would get looking only at the numbers from in [1,n] only once. If the difference of the counts is positive, we add that bit to our answer.

```python3
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        n = len(nums) - 1
		# gives the maximum binary length of a number in [1,n]
        max_length = n.bit_length()
        ans = 0
        
        for j in range(max_length):
			# moves the bits in 1 j positons to the left
			# thus mask has a 1 in the j-th position and 0s everywhere else
            mask = 1 << j
            count = 0
            for i in range(n + 1):
				# if nums[i] has a 1 in the j-th position
                if nums[i] & mask > 0:
                    count += 1
				# if i has a 1 in the j-th position
                if i & mask > 0:
                    count -= 1
			#if we found extra 1s in the j-th position add that bit to ans
            if count > 0:
                ans |= mask
                    
        return ans
```

**Time:** O(n log(n))
**Space:** O(1)

## 141. Roman to Integer
**Reference:** https://leetcode.com/problems/roman-to-integer/solutions/6537/my-straightforward-python-solution/

**Description:** Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

**Constraints:** 
1. 1 <= s.length <= 15
2. s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').
3. It is guaranteed that s is a valid roman numeral in the range [1, 3999].

**Examples:** 
```python3
s = "III" #=> 3
s = "LVIII" #=> 58
s = "MCMXCIV" #=> 1994
```

**Hint:** 
Create a map between characters and values. 
Iterate over the input string keeping track of the curr and next character. If next is less than or equal to current, add hashMap[curr] to output total. Else (next is > curr) subtract curr from output value

```python3
class Solution:
    def romanToInt(self, s):
        roman = {'M': 1000,'D': 500 ,'C': 100,'L': 50,'X': 10,'V': 5,'I': 1}
        z = 0
        for i in range(0, len(s) - 1):
            if roman[s[i]] < roman[s[i+1]]:
                z -= roman[s[i]]
            else:
                z += roman[s[i]]
        return z + roman[s[-1]]
```

**Time:** O(n)
**Space:** O(1)

## 142. Palindrome Number
**Reference:** https://leetcode.com/problems/palindrome-number/solutions/785314/python-3-1-solution-is-89-20-faster-2nd-is-99-14-faster-explanation-added/

**Description:** Given an integer x, return true if x is a palindrome, and false otherwise.

Follow up: Could you solve it without converting the integer to a string?

**Constraints:** 
-2^31 <= x <= 2^31 - 1

**Examples:** 
```python3
x = 121 #=> true
x = -121 #=> false
x = 10 #=> false
```

**Hint:** 
if input number is negative or last digit is 10 return False

while input number is > 0
1. Build the reversed number digit by digit. Use newNum *= 10 to account for digit shifting. Add inputNum % 10 to get current digit. 
2. Use inputNum = inputNum // 10 to remove least significant bit
After the loop compare the newNum with the inputNum.

```python3
class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: return False
        inputNum = x
        newNum = 0
        while x > 0:
            newNum = newNum * 10 + x % 10
            x = x // 10
        return newNum == inputNum
```

**Time:** O(n)
**Space:** O(1)

## 143. Random Pick with Weight
**Reference:** 
https://www.educative.io/answers/what-is-the-weighted-random-selection-algorithm
https://leetcode.com/problems/random-pick-with-weight/solutions/884261/array-binary-search-faster-than-97/
 
**Description:** You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.

You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).

For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).

**Constraints:** 
1 <= w.length <= 10^4
1 <= w[i] <= 10^5
pickIndex will be called at most 10^4 times.

**Examples:** 
```python3
["Solution","pickIndex"]
[[[1]],[]] #=> [null,0]

["Solution","pickIndex","pickIndex","pickIndex","pickIndex","pickIndex"]
[[[1,3]],[],[],[],[],[]] #=> [null,1,1,1,1,0]
```

**Hint:** 
1. Use the input array to create an array of prefix sums (e.g., running total of input array)

input = [1,2,3]
prefix sums = [1,3,6]

2. generate a random number between 0 and sum(nums) - 1
3. Find the smallest index of nums that corresponds to the prefix array that is greater than the random number (use binary search) O(logn)

```python3
class Solution:
    def __init__(self, w):
        self.weights = []
        self.sum = 0
        for weight in w:
            self.sum += weight
            self.weights.append(self.sum)

    def pickIndex(self):
        index = random.randint(0, self.sum - 1)
        left, right = 0, len(self.weights) - 1
        while left < right:
            mid = left + (right - left) // 2
            if self.weights[mid] <= index:
                left = mid + 1
            else:
                right = mid
        return left
```

**Time:** O(n)
**Space:** O(n)

## 144. Pow(x, n)
**Reference:** https://leetcode.com/problems/powx-n/solutions/19560/shortest-python-guaranteed/
 
**Description:** Implement pow(x, n), which calculates x raised to the power n (i.e., x^n).

**Constraints:** 
1. -100.0 < x < 100.0
2. -2^31 <= n <= 2^31-1
3. n is an integer.
4. Either x is not zero or n > 0.
5. -10^4 <= x^n <= 10^4

**Examples:** 
```python3
x = 2.00000, n = 10 #=> 1024.00000
x = 2.10000, n = 3 #=> 9.26100
x = 2.00000, n = -2 #=> 0.25000
```

**Hint:** 
Recursive approach is probably easiet to understand

Base case: if exponent == 0, return 1

if exponent is negative return 1 / positive version: x^-1 = 1/x

if exponent is odd return number * positive version - 1: 3^3 = 3 * 3^2

if exponent is even, return positive version / 2: 3^2 = 3*3^1

```python3
class Solution:
    def myPow(self, x, n):
        if not n:
            return 1
        if n < 0:
            return 1 / self.myPow(x, -n)
        if n % 2:
            return x * self.myPow(x, n-1)
        return self.myPow(x*x, n/2)
```

**Time:** O(logn)
**Space:** O(1)

## 145. Reverse Integer
**Reference:** https://leetcode.com/problems/reverse-integer/description/
 
**Description:** Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-2^31, 2^31 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

**Constraints:** 
-2^31 <= x <= 2^31 - 1

**Examples:** 
```python3
x = 123 #=> 321
x = -123 #=> -321
x = 120 #=> 21
```

**Hint:** 
The basic idea is while the input number > 0, add the digit with rev= rev*10 + x % 10 and decrement with x = x // 10.  

You can check for overflow confirming that previous number != currentNumber: (rev - x % 10) / 10 != prevNum

```python3
class Solution:
# https://leetcode.com/problems/reverse-integer/solutions/408697/two-python-solutions-and-explanation-of-python-modulo-and-int-division-differences-from-c-java/
    def reverse(self, x: int) -> int:
        reverse = 0
        max_int = pow(2, 31)-1
        min_int = pow(-2, 31)
        while x != 0:   
            # Python modulo does not work the same as c or java. It always returns the same
            # sign as the divisor and rounds towards negative infinit. Also // rounds towards negative infinity not 0 as in C so this also
            # behaves differently. Python 3.7 added a math.remainder(), but leet code is
            # running a python version prior to this (at least at the time of writing). Since the C 'remainder' behavior is desirable for
            # this problem, the following code emulates it. 
            #
            # See https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation and
			# http://python-history.blogspot.com/2010/08/why-pythons-integer-division-floors.html
            pop = x % 10 if x >= 0 else (abs(x) % 10)*-1
            x = x // 10 if x >=0 else math.ceil(x / 10)
            if (reverse > max_int//10) or (reverse == max_int // 10 and pop > 7):
                return 0
            if (reverse < math.ceil(min_int / 10)) or (reverse == math.ceil(min_int / 10) and pop < -8):
                return 0
            reverse = reverse * 10 + pop
        return reverse
```
**Time:** O(n)
**Space:** O(1)

```java
// https://leetcode.com/problems/reverse-integer/solutions/4056/very-short-7-lines-and-elegant-solution/
  public int reverse(int x) {
        int prevRev = 0 , rev= 0;
        while( x != 0){
            rev= rev*10 + x % 10;
            if((rev - x % 10) / 10 != prevRev){
                return 0;
            }
            prevRev = rev;
            x= x/10;
        }
        return rev;
    }
```

**Time:** O(n)
**Space:** O(1)

## 146. K Closest Points to Origin
**Reference:** https://leetcode.com/problems/k-closest-points-to-origin/solutions/294389/easy-to-read-python-min-heap-solution-beat-99-python-solutions/
 
**Description:** Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)^2 + (y1 - y2)^2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

**Constraints:** 
1 <= k <= points.length <= 10^4
-10^4 <= xi, yi <= 10^4

**Examples:** 
```python3
points = [[1,3],[-2,2]], k = 1 #=> [[-2,2]]
```

![image](https://github.com/will4skill/algo-review/assets/10373005/8f86c747-cffb-43db-8b09-955a3c1e4f36)

```python3
points = [[3,3],[5,-1],[-2,4]], k = 2 #=> [[3,3],[-2,4]]
```

**Hint:** 
Iterate over each point, pushing them into a max heap unless heap size is K

If the heap size reaches K, push, then pop largest element.

At the end, the max heap will only contain the closest k element. Return contents of the heap

Note: you could use quick select (avg O(n) worst case O(n^2))

```python3
class Solution(object):
    def kClosest(self, points, k):
        heap = []
        for (x, y) in points:
            dist = -(x*x + y*y)
            if len(heap) == k:
                heapq.heappushpop(heap, (dist, x, y))
            else:
```

**Time:** O(N * logK)
**Space:** O(K)

## 147. Task Scheduler
**Reference:** https://leetcode.com/problems/task-scheduler/solutions/104511/Java-Solution-PriorityQueue-and-HashMap/
 
**Description:** Given a characters array tasks, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer n that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least n units of time between any two same tasks.

Return the least number of units of times that the CPU will take to finish all the given tasks.

**Constraints:** 
1 <= task.length <= 10^4
tasks[i] is upper-case English letter.
The integer n is in the range [0, 100].

**Examples:** 
```python3
tasks = ["A","A","A","B","B","B"], n = 2 #=> 8
```

```python3
tasks = ["A","A","A","B","B","B"], n = 0 #=> 6
```

```python3
tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2 #=> 16
```

**Hint:** 
Greedy
1. Create a map to track the frequency of each task
2. Sort it from min to max
3. Pick elements in descending order, one by one. At each step, decease the idle time by min(f-max - 1, f). Where f is the current frequency
4. Return busy + idle slots.

```python3
class Solution(object):
    def leastInterval(self, tasks, n):
        if n == 0: return len(tasks)
        
        task_to_count = {}
        for c in tasks:
            task_to_count[c] = task_to_count.get(c, 0) + 1
        
        queue = []
        for count in task_to_count.values():
            queue.append(-count) # Max heap?
        
        heapq.heapify(queue)
        
        cooldown = {}
        curr_time = 0
        while queue or cooldown:
            if curr_time - n - 1 in cooldown:
                heapq.heappush(queue, cooldown.pop(curr_time - n - 1))
            
            if queue:
                count = -heapq.heappop(queue) - 1
                if count > 0:
                    cooldown[curr_time] = -count
            
            curr_time += 1
        
        return curr_time
```

**Time:** O(Ntotal), where Ntotal is the number of tasks to complete
**Space:** O(1) to keep array of 26 elements

## 148. Top K Frequent Words
**Reference:** https://leetcode.com/problems/top-k-frequent-words/solutions/108346/my-simple-java-solution-using-hashmap-priorityqueue-o-nlogk-time-o-n-space/
 
**Description:** Given an array of strings words and an integer k, return the k most frequent strings.

Return the answer sorted by the frequency from highest to lowest. Sort the words with the same frequency by their lexicographical order.

Follow-up: Could you solve it in O(n log(k)) time and O(n) extra space?

**Constraints:** 
1. 1 <= words.length <= 500
2. 1 <= words[i].length <= 10
3. words[i] consists of lowercase English letters.
4. k is in the range [1, The number of unique words[i]]

**Examples:** 
```python3
words = ["i","love","leetcode","i","love","coding"], k = 2 #=> ["i","love"]
```

```python3
words = ["the","day","is","sunny","the","the","the","sunny","is","is"], k = 4 #=> ["the","is","sunny","day"]
```

**Hint:** 
Use a hash map to store the count of each word

Insert the word into your custom priority queue. If the counts of two words are equal, string compare the keys

```python3
class Solution(object):
    def topKFrequent(self, words, k):
        if k == 0: return []
        
        word_count = {}
        # for word in words:
        #     if word in word_count:
        #         word_count[word] += 1
        #     else:
        #         word_count[word] = 1
        for word in words:
            word_count[word] = word_count.get(word, 1) + 1
        
        import heapq
        heap = [(-freq, word) for word, freq in word_count.items()]
        heapq.heapify(heap)
        
        result = []
        for _ in range(k):
            result.append(heapq.heappop(heap)[1])
        
        return result
```

**Time:** O(nlogk)
**Space:** O(n)

## 149. Find K Closest Elements
**Reference:** https://leetcode.com/problems/find-k-closest-elements/description/
 
**Description:** Given a sorted integer array arr, two integers k and x, return the k closest integers to x in the array. The result should also be sorted in ascending order.

An integer a is closer to x than an integer b if:
1. |a - x| < |b - x|, or
2. |a - x| == |b - x| and a < b

**Constraints:** 
1. 1 <= k <= arr.length
2. 1 <= arr.length <= 10^4
3. arr is sorted in ascending order.
4. -10^4 <= arr[i], x <= 10^4

**Examples:** 
```python3
arr = [1,2,3,4,5], k = 4, x = 3 #=> [1,2,3,4]
arr = [1,2,3,4,5], k = 4, x = -1 #=> [1,2,3,4]
```

**Hint:** 
You can use a priority queue O(nlog(k)) but the preferred solution is binary search

PQ version: 
1. iterate over arr, filling min heap
2. if a remaining element is closer than the min element, pop and replace with remaining element
3. Pop all elements from pq and return

bin search version: the key is to find the starting element
1. Return [start, start+k]
2. let hi = len(arr)-k and lo = 0
3. Check: if distance from mid to x is greater than distance from mid+k to x, disgard the lower half (lo = mid + 1)

```python3
# https://leetcode.com/problems/find-k-closest-elements/solutions/2636647/java-explained-in-detail-binary-search-two-pointers-priority-queue/
import heapq
class Solution(object):
    # Approach:
    # Using a min heap priority queue, add all the smallest integers up to k integers.
    # Then, traverse the 'arr' array will replacing the priority queue with integer closer to x.
    def findClosestElements(self, arr, k, x):
        min_heap = []
        for num in arr:
            if len(min_heap) < k: # Fill heap 
                heapq.heappush(min_heap, num)
            else: # replace the min if you find a better option in the remaining elements
                if abs(min_heap[0] - x)> abs(num - x):
                    heapq.heappop(min_heap)
                    heapq.heappush(min_heap, num)
        
        ans = []
        while min_heap: # pop everything. the heap will maintain ascending order
            ans.append(heapq.heappop(min_heap))
        
        return ans
```

**Time:** O(nlogk)
**Space:** O(k)

```python3
# https://leetcode.com/problems/find-k-closest-elements/solutions/133604/clean-o-logn-solution-in-python/
import heapq
class Solution(object):
    def findClosestElements(self, arr, k, x):
        lo, hi = 0, len(arr)-k
        while lo<hi:
            mid = (lo + hi)//2
            if x-arr[mid]>arr[mid+k]-x:
                lo = mid + 1
            else:
                hi = mid
        return arr[lo:lo+k]
```

**Time:** O(logN + k)
**Space:** O(k)

## 150. Kth Largest Element in an Array
**Reference:** https://leetcode.com/problems/kth-largest-element-in-an-array/description/
 
**Description:** Given an integer array nums and an integer k, return the kth largest element in the array.

Note that it is the kth largest element in the sorted order, not the kth distinct element.

Can you solve it without sorting?

**Constraints:** 
1 <= k <= nums.length <= 10^5
-10^4 <= nums[i] <= 10^4

**Examples:** 
```python3
nums = [3,2,1,5,6,4], k = 2 #=> 5
nums = [3,2,3,1,2,4,5,5,6], k = 4 #=> 4
```

**Hint:** 
Iterate over all the elemets, inserting into a min heap. If the size reaches k, push then pop the next value to maintain size k.
Return min value.

Note: the preferred solution seems to be quick select O(n) avg O(n^2) worst case with space O(1)

```python3
# https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/60294/solution-explained/
class Solution:
    def findKthLargest(self, nums, k):
        min_heap = []
        for num in nums:
            heapq.heappush(min_heap, num)
            if len(min_heap) > k:
                heapq.heappop(min_heap)
        return min_heap[0]
```

**Time:** O(N lg K)
**Space:** O(K)

```python3
# https://leetcode.com/problems/kth-largest-element-in-an-array/solutions/3906260/100-3-approaches-video-heap-quickselect-sorting/
class Solution:
    def findKthLargest(self, nums, k):
        left, right = 0, len(nums) - 1
        while True:
            pivot_index = random.randint(left, right)
            new_pivot_index = self.partition(nums, left, right, pivot_index)
            if new_pivot_index == len(nums) - k:
                return nums[new_pivot_index]
            elif new_pivot_index > len(nums) - k:
                right = new_pivot_index - 1
            else:
                left = new_pivot_index + 1

    def partition(self, nums, left, right, pivot_index):
        pivot = nums[pivot_index]
        nums[pivot_index], nums[right] = nums[right], nums[pivot_index]
        stored_index = left
        for i in range(left, right):
            if nums[i] < pivot:
                nums[i], nums[stored_index] = nums[stored_index], nums[i]
                stored_index += 1
        nums[right], nums[stored_index] = nums[stored_index], nums[right]
        return stored_index
```

**Time:** O(N) avg, O(N^2) worst case 
**Space:** O(1)

## 151. Find Median from Data Stream
**Reference:** https://leetcode.com/problems/find-median-from-data-stream/solutions/74047/JavaPython-two-heap-solution-O(log-n)-add-O(1)-find/
 
**Description:** The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

1. For example, for arr = [2,3,4], the median is 3.
2. For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
   
Implement the MedianFinder class:
1. MedianFinder() initializes the MedianFinder object.
2. void addNum(int num) adds the integer num from the data stream to the data structure.
3. double findMedian() returns the median of all elements so far. Answers within 10^-5 of the actual answer will be accepted.

Follow up:
1. If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
2. If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?

**Constraints:** 
1. -10^5 <= num <= 10^5
2. There will be at least one element in the data structure before calling findMedian.
3. At most 5 * 10^4 calls will be made to addNum and findMedian.

**Examples:** 
```python3
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []] #=> [null, null, null, 1.5, null, 2.0]
```

**Hint:** 
1. Use a max heap to store the smaller half and a min heap to store the larger half
2. addNum: if number of elements is even, insert number into min heap and pop from max heap. If number of elements is odd, insert into max heap, and poll element from min heap. Flip even boolean.
3. findMedian: If number of elements is even return min.poll() max.pol()  / 2 else return max.poll()

```python3
class MedianFinder:
    def __init__(self):
        self.small = []  # the smaller half of the list, max heap (invert min-heap)
        self.large = []  # the larger half of the list, min heap

    def addNum(self, num):
        if len(self.small) == len(self.large):
            heappush(self.large, -heappushpop(self.small, -num))
        else:
            heappush(self.small, -heappushpop(self.large, num))

    def findMedian(self):
        if len(self.small) == len(self.large):
            return float(self.large[0] - self.small[0]) / 2.0
        else:
            return float(self.large[0])
```

**Time:** O(log n) add, O(1) find
**Space:** O(n)

## 152. Merge k Sorted Lists
**Reference:** https://leetcode.com/problems/merge-k-sorted-lists/solutions/354814/Java-Heap-Solution/
 
**Description:** You are given an array of k linked-lists lists, each linked-list is sorted in ascending order. Merge all the linked-lists into one sorted linked-list and return it.

**Constraints:** 
1. k == lists.length
2. 0 <= k <= 10^4
3. 0 <= lists[i].length <= 500
4. -10^4 <= lists[i][j] <= 10^4
5. lists[i] is sorted in ascending order.
6. The sum of lists[i].length will not exceed 10^4.

**Examples:** 
```python3
lists = [[1,4,5],[1,3,4],[2,6]] #=> [1,1,2,3,4,4,5,6]
lists = [] #=> []
lists = [[]] #=> []
```

**Hint:** 
1. Create a custom comparator to compare list node.val
2. Add all of your list nodes to your heap
3. Create a new sentenel list node
4. Poll entire heap while creating a LL chain 
5. Return sentinel.next 

```python3
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for linked_list in lists:
            while linked_list:
                heapq.heappush(heap, linked_list.val)
                linked_list = linked_list.next
        
        sentinel = ListNode(-1)
        cur = sentinel
        while heap:
            cur.next = ListNode(heapq.heappop(heap))
            cur = cur.next
        
        return sentinel.next
```

**Time:** O(N log k)
**Space:** O(N)

## 153. Smallest Range Covering Elements from K Lists
**Reference:** https://leetcode.com/problems/smallest-range-covering-elements-from-k-lists/solutions/104893/java-code-using-priorityqueue-similar-to-merge-k-array/
 
**Description:** You have k lists of sorted integers in non-decreasing order. Find the smallest range that includes at least one number from each of the k lists.

We define the range [a, b] is smaller than range [c, d] if b - a < d - c or a < c if b - a == d - c.

**Constraints:** 
1. nums.length == k
2. 1 <= k <= 3500
3. 1 <= nums[i].length <= 50
4. -10^5 <= nums[i][j] <= 10^5
5. nums[i] is sorted in non-decreasing order.
   
**Examples:** 
```python3
nums = [[4,10,15,24,26],[0,9,12,20],[5,18,22,30]] #=> [20,24]
nums = [[1,2,3],[1,2,3],[1,2,3]] #=> [1,1]
```

**Hint:** 
1. Create a min heap
2. Insert the minimum value from each list into the heap and note the largest of those min values as maxVal. Let range be a large value. 
3. While the size of the heap is the number of different lists, poll the minimum value as curr.
4. If the maxVal - curr (current min) improves the range. Update the range to maxVal - curr. Update start to curr.val and update end to maxVal.
5. if curr.idx + 1 < is a valid list element (the next element is not out of bounds), insert the next element in the curr list input the heap and update the max if that new value is greater than the existing max.
6. Return the range

```python3
class Solution:
    def smallestRange(self, nums):
        pq = []
        max_val = float('-inf')
        for i, row in enumerate(nums):
            if row:
                val = row.pop(0)
                heapq.heappush(pq, (val, i))
                max_val = max(max_val, val)
        
        start, end = float('-inf'), float('inf')
        while len(pq) == len(nums):
            val, i = heapq.heappop(pq)
            if max_val - val < end - start:
                start, end = val, max_val
            if nums[i]:
                new_val = nums[i].pop(0)
                heapq.heappush(pq, (new_val, i))
                max_val = max(max_val, new_val)
        
        return [start, end]
```

**Time:** O(n * log(m)) Heapifying m elements takes O(log(m)) time, n is the total number of elements in all lists, m is the total number of lists
**Space:** O(m)

## 154. Implement Trie (Prefix Tree)
**Reference:** https://leetcode.com/problems/implement-trie-prefix-tree/solutions/58989/my-python-solution/
 
**Description:** A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:
1. Trie() Initializes the trie object.
2. void insert(String word) Inserts the string word into the trie.
3. boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
4. boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

**Constraints:** 
1. 1 <= word.length, prefix.length <= 2000
2. word and prefix consist only of lowercase English letters.
3. At most 3 * 10^4 calls in total will be made to insert, search, and startsWith.
   
**Examples:** 
```python3
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]] #=> [null, null, true, false, true, null, true]
```

**Hint:** 
1. The nodes just need properties of: a map of children, isWord boolean
2. insert: for each char in word, if the current node's children do not include the curr char, insert it as a new Node. Either way, increment: node = node.children[char]. At the end, set isWord to true
3. search: for each char in word, if the current node's children do not include the char return false. Either way, increment as above. At the end, return isWord
4. startsWith: for each char in word, if the current node's children do not include the char return false. Either way, increment as above. At the end, return true.

```python3
class TrieNode:
    # Initialize your data structure here.
    def __init__(self):
        self.word=False
        self.children={}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    # @param {string} word
    # @return {void}
    # Inserts a word into the trie.
    def insert(self, word: str) -> None:
        node=self.root
        for i in word:
            if i not in node.children:
                node.children[i]=TrieNode()
            node=node.children[i]
        node.word=True       

    # @param {string} word
    # @return {boolean}
    # Returns if the word is in the trie.
    def search(self, word: str) -> bool:
        node=self.root
        for i in word:
            if i not in node.children:
                return False
            node=node.children[i]
        return node.word

    # @param {string} prefix
    # @return {boolean}
    # Returns if there is any word in the trie
    # that starts with the given prefix.
    def startsWith(self, prefix: str) -> bool:
        node=self.root
        for i in prefix:
            if i not in node.children:
                return False
            node=node.children[i]
        return True   
```

**Time:** insert: O(n), search: O(n), startsWith: O(n)
**Space:** insert: O(n), search: O(1), startsWith: O(1)

