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
