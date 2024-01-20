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
