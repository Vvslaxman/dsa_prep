class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        new_node = Node(data)
        new_node.next = self.head
        self.head = new_node
    
    def print_list(self):
        temp = self.head
        while temp:
            print(temp.data, end=" -> ")
            temp = temp.next
        print("None")

    def find_middle(self):  # Tortoise-Hare approach
        slow, fast = self.head, self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow.data if slow else None

    def detect_cycle(self):  # Floyd’s Cycle Detection
        slow, fast = self.head, self.head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow == fast:
                return True
        return False

    def reverse_iterative(self):
        prev, curr = None, self.head
        while curr:
            nxt = curr.next
            curr.next = prev
            prev = curr
            curr = nxt
        self.head = prev

    def reverse_recursive(self, head):
        if not head or not head.next:
            return head
        new_head = self.reverse_recursive(head.next)
        head.next.next = head
        head.next = None
        return new_head

# Time & Space Complexity:
# - Insertion: O(1), O(1)
# - Find Middle: O(N), O(1)
# - Detect Cycle: O(N), O(1)
# - Reverse Iterative: O(N), O(1)
# - Reverse Recursive: O(N), O(N) (due to recursion stack)

# --- Kadane’s Algorithm (Maximum Subarray Sum) ---
def max_subarray_sum(arr):
    max_sum = float('-inf')
    curr_sum = 0
    for num in arr:
        curr_sum = max(num, curr_sum + num)
        max_sum = max(max_sum, curr_sum)
    return max_sum
# Time: O(N), Space: O(1)

# --- Two Sum ---
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
# Time: O(N), Space: O(N)

# --- Longest Palindromic Substring ---
def longest_palindromic_substring(s):
    def expand_around_center(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left + 1:right]
    
    longest = ""
    for i in range(len(s)):
        odd = expand_around_center(i, i)
        even = expand_around_center(i, i + 1)
        longest = max(longest, odd, even, key=len)
    return longest
# Time: O(N^2), Space: O(1)

# --- Sliding Window Maximum ---
from collections import deque

def sliding_window_max(nums, k):
    q = deque()
    result = []
    for i in range(len(nums)):
        while q and q[0] < i - k + 1:
            q.popleft()
        while q and nums[q[-1]] < nums[i]:
            q.pop()
        q.append(i)
        if i >= k - 1:
            result.append(nums[q[0]])
    return result
# Time: O(N), Space: O(K)

# --- Longest Substring Without Repeating Characters ---
def longest_unique_substring(s):
    char_map = {}
    left = max_length = 0
    for right, char in enumerate(s):
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1
        char_map[char] = right
        max_length = max(max_length, right - left + 1)
    return max_length
# Time: O(N), Space: O(N)

# More problems will be added soon...

# --- Find the kth Largest/Smallest Element ---
import heapq

def kth_largest(nums, k):
    return heapq.nlargest(k, nums)[-1]

def kth_smallest(nums, k):
    return heapq.nsmallest(k, nums)[-1]
# Time: O(N log K), Space: O(K)

# --- Merge Two Sorted Arrays Efficiently ---
def merge_sorted_arrays(arr1, arr2):
    i, j, merged = 0, 0, []
    while i < len(arr1) and j < len(arr2):
        if arr1[i] < arr2[j]:
            merged.append(arr1[i])
            i += 1
        else:
            merged.append(arr2[j])
            j += 1
    merged.extend(arr1[i:])
    merged.extend(arr2[j:])
    return merged
# Time: O(N+M), Space: O(N+M)

# --- Check if a String is a Palindrome Efficiently ---
def is_palindrome(s):
    left, right = 0, len(s) - 1
    while left < right:
        if s[left] != s[right]:
            return False
        left += 1
        right -= 1
    return True
# Time: O(N), Space: O(1)
