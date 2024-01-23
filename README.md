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

```java
class Solution {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; i++) {
            res[i] = res[i - 1] * nums[i - 1];
        }
        int right = 1;
        for (int i = n - 1; i >= 0; i--) {
            res[i] *= right;
            right *= nums[i];
        }
        return res;
    }
}
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

```java
public List<List<Integer>> combinationSum(int[] nums, int target) {
    List<List<Integer>> list = new ArrayList<>();
    Arrays.sort(nums);
    backtrack(list, new ArrayList<>(), nums, target, 0);
    return list;
}

private void backtrack(List<List<Integer>> list, List<Integer> tempList, int [] nums, int remain, int start){
    if(remain < 0) return;
    else if(remain == 0) list.add(new ArrayList<>(tempList));
    else{ 
        for(int i = start; i < nums.length; i++){
            tempList.add(nums[i]);
            backtrack(list, tempList, nums, remain - nums[i], i); // not i + 1 because we can reuse same elements
            tempList.remove(tempList.size() - 1);
        }
    }
}
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

```java
public void sortColors(int[] nums) {
    // 1-pass
    int p1 = 0, p2 = nums.length - 1, index = 0;
    while (index <= p2) {
        if (nums[index] == 0) {
            nums[index] = nums[p1];
            nums[p1] = 0;
            p1++;
        }
        if (nums[index] == 2) {
            nums[index] = nums[p2];
            nums[p2] = 2;
            p2--;
            index--;
        }
        index++;
    }
}
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

```java
public class Solution {
    public int findMaxLength(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        map.put(0, -1);
        int maxlen = 0, count = 0;
        for (int i = 0; i < nums.length; i++) {
            count = count + (nums[i] == 1 ? 1 : -1);
            if (map.containsKey(count)) {
                maxlen = Math.max(maxlen, i - map.get(count));
            } else {
                map.put(count, i);
            }
        }
        return maxlen;
    }
}
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

```java
public class Solution {
    public int subarraySum(int[] nums, int k) {
        int count = 0, sum = 0;
        HashMap < Integer, Integer > map = new HashMap < > ();
        map.put(0, 1);
        for (int i = 0; i < nums.length; i++) {
            sum += nums[i];
            if (map.containsKey(sum - k))
                count += map.get(sum - k);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        return count;
    }
}
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

```java
public int[] asteroidCollision(int[] asteroids) {
        Stack<Integer> s = new Stack<>();
        for(int i: asteroids){
            if(i > 0){
                s.push(i);
            }else{// i is negative
                while(!s.isEmpty() && s.peek() > 0 && s.peek() < Math.abs(i)){
                    s.pop();
                }
                if(s.isEmpty() || s.peek() < 0){
                    s.push(i);
                }else if(i + s.peek() == 0){
                    s.pop(); //equal
                }
            }
        }
        int[] res = new int[s.size()];   
        for(int i = res.length - 1; i >= 0; i--){
            res[i] = s.pop();
        }
        return res;
    }
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

```java
public class Solution {
	public int calculate(String s) {
	    int len;
	    if(s==null || (len = s.length())==0) return 0;
	    Stack<Integer> stack = new Stack<Integer>();
	    int num = 0;
	    char sign = '+';
	    for(int i=0;i<len;i++){
	        if(Character.isDigit(s.charAt(i))){
	            num = num*10+s.charAt(i)-'0';
	        }
	        if((!Character.isDigit(s.charAt(i)) &&' '!=s.charAt(i)) || i==len-1){
	            if(sign=='-'){
	                stack.push(-num);
	            }
	            if(sign=='+'){
	                stack.push(num);
	            }
	            if(sign=='*'){
	                stack.push(stack.pop()*num);
	            }
	            if(sign=='/'){
	                stack.push(stack.pop()/num);
	            }
	            sign = s.charAt(i);
	            num = 0;
	        }
	    }
	
	    int re = 0;
	    for(int i:stack){
	        re += i;
	    }
	    return re;
	}
}
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

```cpp
class Solution {
public:
    int trap(int A[], int n) {
        int left=0; int right=n-1;
        int res=0;
        int maxleft=0, maxright=0;
        while(left<=right){
            if(A[left]<=A[right]){
                if(A[left]>=maxleft) maxleft=A[left];
                else res+=maxleft-A[left];
                left++;
            }
            else{
                if(A[right]>=maxright) maxright= A[right];
                else res+=maxright-A[right];
                right--;
            }
        }
        return res;
    }
};
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

```cpp
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.length(), longest = 0;
        stack<int> st;
        for (int i = 0; i < n; i++) {
            if (s[i] == '(') st.push(i);
            else {
                if (!st.empty()) {
                    if (s[st.top()] == '(') st.pop();
                    else st.push(i);
                }
                else st.push(i);
            }
        }
        if (st.empty()) longest = n;
        else {
            int a = n, b = 0;
            while (!st.empty()) {
                b = st.top(); st.pop();
                longest = max(longest, a-b-1);
                a = b;
            }
            longest = max(longest, a);
        }
        return longest;
    }
};
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
